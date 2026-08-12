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

  await page.goto(`${BASE}/app/create/text-to-video`, { waitUntil: "domcontentloaded" });
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

  // The workspace may already be showing a COMPLETED job from an earlier run
  // (it auto-selects the newest). Sampling immediately would see its Download
  // button and read the run as finished before it started — so wait for the
  // canvas to switch to the new job's progress view first.
  await page
    .locator('main [role="status"]')
    .first()
    .waitFor({ state: "visible", timeout: 15_000 });

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
  await page.goto(`${BASE}/app/generations`, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(800);
  const cards = await page.locator('main a[href^="/app/generations/"]').count();
  check("history lists the generation", cards >= 1, `${cards} card(s)`);

  await page.locator('main a[href^="/app/generations/"]').first().click();
  await page.waitForURL(/\/app\/generations\/[^/]+/, { timeout: 15_000 });
  await page.waitForLoadState("networkidle");
  check("detail page loads by id", true);

  // ── 6. Its output is in the media library ──────────────────────────────
  console.log("\n6. Media library");
  await page.goto(`${BASE}/app/media`, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(800);
  const assets = await page.locator('main button[aria-label^="Select "]').count();
  check("library shows the generated asset", assets >= 1, `${assets} asset(s)`);

  // ── 7. Workflow switching drives the UI from metadata ──────────────────
  console.log("\n7. Workflow-driven UI");
  await page.goto(`${BASE}/app/create/music`, { waitUntil: "domcontentloaded" });
  check(
    "Music hides the aspect-ratio section",
    (await page.locator('[role="group"][aria-label="Aspect ratio"]').count()) === 0,
  );
  const musicDurations = await page
    .locator('[role="group"][aria-label="Duration"] button')
    .allInnerTexts();
  check(
    "Music is chosen in minutes (CR-009)",
    JSON.stringify(musicDurations) ===
      JSON.stringify(["1 min", "2 min", "3 min", "4 min", "5 min"]),
    musicDurations.join("/"),
  );

  await page.goto(`${BASE}/app/create/video-to-video`, { waitUntil: "domcontentloaded" });
  const bodyText = await page.locator('aside[aria-label="Generation settings"]').innerText();
  check("Video to Video shows a required source video", bodyText.includes("INPUT VIDEO"));
  check(
    "…and an OPTIONAL reference image",
    bodyText.includes("REFERENCE IMAGE") && bodyText.includes("optional"),
  );
  check(
    "Video to Video duration is automatic (CR-006)",
    (await page.locator('[data-testid="auto-duration"]').innerText()).includes(
      "Same as source video",
    ) && (await page.locator('[role="group"][aria-label="Duration"]').count()) === 0,
  );

  await page.goto(`${BASE}/app/create/extend-video`, { waitUntil: "domcontentloaded" });
  const extendDurations = await page
    .locator('[role="group"][aria-label="Duration"] button')
    .allInnerTexts();
  check(
    "Extend offers 5/10/15/30/60 (CR-008)",
    JSON.stringify(extendDurations) === JSON.stringify(["5s", "10s", "15s", "30s", "60s"]),
    extendDurations.join("/"),
  );

  // ── 8. Settings preservation across a workflow switch ──────────────────
  console.log("\n8. Settings preservation");
  await page.goto(`${BASE}/app/create/text-to-video`, { waitUntil: "domcontentloaded" });
  await page.locator('[role="group"][aria-label="Duration"] button', { hasText: "15s" }).click();
  await page.goto(`${BASE}/app/create/image-to-video`, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(400);
  const selected = await page
    .locator('[role="group"][aria-label="Duration"] button[aria-pressed="true"]')
    .innerText();
  check(
    "unsupported 15s falls back to 5s rather than persisting",
    selected.trim() === "5s",
    selected.trim(),
  );

  // ── 8b. Extend hands its source over ───────────────────────────────────
  //
  // The bug this pins: Extend used to be a bare link, so it opened an empty
  // upload box and the user had to download their own generation and upload
  // it back — which a missing file extension also made impossible.
  console.log("\n8b. Extend receives the result as its source");
  const media = await (await fetch(`${API}/api/v1/media?kind=video&source=generated&limit=1`, {
    headers: { cookie: (await context.cookies()).map((c) => `${c.name}=${c.value}`).join("; ") },
  })).json();
  const sourceAsset = media.items?.[0];
  if (!sourceAsset) {
    check("a generated video exists to extend", false, "no generated video in the library");
  } else {
    check("generated video downloads under a usable filename", sourceAsset.name.includes("."), sourceAsset.name);
    await page.goto(`${BASE}/app/create/extend-video?source=${sourceAsset.id}`, {
      waitUntil: "domcontentloaded",
    });
    // The filled state shows the asset's name and a Remove control; the empty
    // state shows a drop prompt instead.
    const filled = page.getByRole("button", { name: `Remove ${sourceAsset.name}` });
    check(
      "the handed-over source is shown as already provided",
      await filled.isVisible().catch(() => false),
    );
    check(
      "Generate is enabled without a manual upload",
      await page
        .getByRole("button", { name: /generate/i })
        .first()
        .isEnabled()
        .catch(() => false),
    );
  }

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
