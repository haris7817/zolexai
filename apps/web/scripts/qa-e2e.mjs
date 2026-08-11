/**
 * End-to-end verification of the M1 acceptance path.
 *
 * Drives a real browser against the real stack — Next.js, FastAPI, PostgreSQL,
 * Redis, MinIO and a running worker — and walks the exact journey the milestone
 * is accepted on:
 *
 *   open the workspace  →  submit a generation  →  watch SSE progress  →
 *   see the result  →  find it in history  →  find its output in the library
 *
 * Nothing here is stubbed. If the worker is not running, this fails, and that
 * is the point: it verifies the system, not the frontend's opinion of it.
 *
 *   QA_BASE=http://localhost:3000 QA_API=http://localhost:8000 node scripts/qa-e2e.mjs
 */
import { chromium } from "playwright";

const BASE = process.env.QA_BASE || "http://localhost:3000";
const API = process.env.QA_API || "http://localhost:8000";

const results = [];
const check = (name, passed, detail = "") => {
  results.push({ name, passed, detail });
  console.log(`  ${passed ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
};

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await context.newPage();

// A console error during a generation is a real defect, not noise.
const consoleErrors = [];
page.on("pageerror", (error) => consoleErrors.push(error.message));

try {
  // ── 1. Workflow metadata comes from the API ────────────────────────────
  console.log("\n1. Workflow metadata from the API");
  const workflows = await (await fetch(`${API}/api/v1/workflows`)).json();
  check("API serves six workflows", workflows.workflows.length === 6);

  await page.goto(`${BASE}/app/create/text-to-video`, { waitUntil: "networkidle" });
  check(
    "workspace renders the workflow served by the API",
    (await page.locator("main h1").innerText()) === "Text to Video",
  );

  const durations = await page
    .locator('[role="group"][aria-label="Duration"] button')
    .allInnerTexts();
  check(
    "durations match the definition",
    JSON.stringify(durations) ===
      JSON.stringify(workflows.workflows.find((w) => w.id === "text-to-video").supported_durations),
    durations.join("/"),
  );

  // ── 2. Validation before submission ────────────────────────────────────
  console.log("\n2. Client validation");
  const generate = page.locator('button[type="submit"]').last();
  check("Generate is disabled with an empty prompt", await generate.isDisabled());

  // ── 3. Submit and watch it stream ──────────────────────────────────────
  console.log("\n3. Submit → SSE progress → result");
  await page.locator("#zx-prompt").fill("A cinematic drone shot over a neon city at dusk");
  await page.waitForTimeout(300);
  check("Generate enables once the prompt is valid", await generate.isEnabled());

  const [submission] = await Promise.all([
    page.waitForResponse(
      (r) => r.url().includes("/api/v1/generations") && r.request().method() === "POST",
    ),
    generate.click(),
  ]);
  check("submission returns 202 immediately", submission.status() === 202, `HTTP ${submission.status()}`);

  const stages = new Set();
  const status = page.locator('main [role="status"]').first();

  // Sample the live region while the worker reports. Distinct stages appearing
  // is the evidence SSE is delivering — a single jump to Completed would mean
  // the UI polled or guessed.
  const deadline = Date.now() + 60_000;
  let completed = false;
  while (Date.now() < deadline) {
    if (await page.locator("text=Download").first().isVisible().catch(() => false)) {
      completed = true;
      break;
    }
    const label = await status.innerText().catch(() => null);
    if (label) stages.add(label.trim());
    await page.waitForTimeout(250);
  }

  check("progress passed through multiple stages", stages.size >= 2, [...stages].join(" → "));
  check("job completed", completed);

  // ── 4. The result is a real stored asset ───────────────────────────────
  console.log("\n4. Result is backed by object storage");
  const media = page.locator("main img, main video, main audio").first();
  const src = await media.getAttribute("src").catch(() => null);
  check("result renders stored media", Boolean(src), src ? new URL(src).pathname : "no media");
  check(
    "media is served from object storage, not the API",
    Boolean(src) && !src.includes("/api/v1/"),
  );

  // ── 5. It appears in history ───────────────────────────────────────────
  console.log("\n5. History");
  await page.goto(`${BASE}/app/generations`, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);
  const cards = await page.locator('main a[href^="/app/generations/"]').count();
  check("history lists the generation", cards >= 1, `${cards} card(s)`);

  await page.locator('main a[href^="/app/generations/"]').first().click();
  await page.waitForLoadState("networkidle");
  check(
    "detail page loads by id",
    page.url().includes("/app/generations/") && !page.url().endsWith("/generations"),
  );

  // ── 6. Its output is in the media library ──────────────────────────────
  console.log("\n6. Media library");
  await page.goto(`${BASE}/app/media`, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);
  const assets = await page.locator('main button[aria-label^="Select "]').count();
  check("library shows the generated asset", assets >= 1, `${assets} asset(s)`);

  // ── 7. Workflow switching drives the UI from metadata ──────────────────
  console.log("\n7. Workflow-driven UI");
  await page.goto(`${BASE}/app/create/music`, { waitUntil: "networkidle" });
  check(
    "Music hides the aspect-ratio section",
    (await page.locator('[role="group"][aria-label="Aspect ratio"]').count()) === 0,
  );
  const musicDurations = await page
    .locator('[role="group"][aria-label="Duration"] button')
    .allInnerTexts();
  check(
    "Music offers its own durations",
    JSON.stringify(musicDurations) === JSON.stringify(["30s", "60s", "120s"]),
    musicDurations.join("/"),
  );

  await page.goto(`${BASE}/app/create/video-to-video`, { waitUntil: "networkidle" });
  const bodyText = await page.locator("aside").innerText();
  check("Video to Video shows a required source video", bodyText.includes("INPUT VIDEO"));
  check(
    "…and an OPTIONAL reference image",
    bodyText.includes("REFERENCE IMAGE") && bodyText.includes("optional"),
  );

  // ── 8. Settings preservation across a workflow switch ──────────────────
  console.log("\n8. Settings preservation");
  await page.goto(`${BASE}/app/create/text-to-video`, { waitUntil: "networkidle" });
  await page.locator('[role="group"][aria-label="Duration"] button', { hasText: "15s" }).click();
  await page.goto(`${BASE}/app/create/image-to-video`, { waitUntil: "networkidle" });
  await page.waitForTimeout(400);
  const selected = await page
    .locator('[role="group"][aria-label="Duration"] button[aria-pressed="true"]')
    .innerText();
  check(
    "unsupported 15s falls back to 5s rather than persisting",
    selected.trim() === "5s",
    selected.trim(),
  );

  // ── 9. No client-visible internals ─────────────────────────────────────
  console.log("\n9. No internal names reach the browser");
  const pageSource = (await page.content()).toLowerCase();
  const leaked = ["ltx", "comfyui", "vast.ai", "pytorch", "postgres", "redis"].filter((n) =>
    pageSource.includes(n),
  );
  check("no provider or infrastructure names in the DOM", leaked.length === 0, leaked.join(", "));

  check("no uncaught page errors", consoleErrors.length === 0, consoleErrors.slice(0, 2).join("; "));
} finally {
  await browser.close();
}

const failed = results.filter((r) => !r.passed);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
