/**
 * Demo-flow QA — walks the guide §7 script and the plan's verification list.
 *
 * Run against a running server:  node scripts/qa-flow.mjs
 */
import { chromium } from "playwright";

const BASE = process.env.QA_BASE || "http://localhost:3213";
const results = [];
const check = (name, pass, detail = "") =>
  results.push({ name, pass, detail });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e).slice(0, 160)));

// ── Step 1–3: empty state, example prompt helper, duration selection ───────
await page.goto(`${BASE}/app/create/text-to-video`, { waitUntil: "networkidle" });

const promptBox = page.locator("#zx-prompt");
check("workspace opens with EMPTY prompt", (await promptBox.inputValue()) === "");

const generateBtn = page
  .locator('aside[aria-label="Generation settings"] button', { hasText: "Generate" })
  .last();
check("Generate disabled on empty prompt", await generateBtn.isDisabled());

await page.getByRole("button", { name: "Use example prompt" }).click();
const filled = await promptBox.inputValue();
check(
  "example prompt fills the field",
  filled.startsWith("A cinematic drone shot flying through a futuristic city"),
  `${filled.length} chars`,
);
check("Generate enabled after prompt", await generateBtn.isEnabled());

// Default duration must be 5s so selecting 10s demonstrates the control
const fiveSel = await page
  .getByRole("button", { name: "5s", exact: true })
  .getAttribute("aria-pressed");
check("default duration is 5s (not pre-set to 10s)", fiveSel === "true");
await page.getByRole("button", { name: "10s", exact: true }).click();
check(
  "selecting 10s updates the control",
  (await page.getByRole("button", { name: "10s", exact: true }).getAttribute("aria-pressed")) === "true",
);

// ── Step 4: simulated pipeline states ─────────────────────────────────────
await generateBtn.click();
const seen = new Set();
const started = Date.now();
// Scope to <main>: the sidebar's active-jobs indicator is ALSO role="status",
// and it precedes the canvas in DOM order.
const stageLabel = page.locator('main [role="status"]').first();
while (Date.now() - started < 12000) {
  const t = await stageLabel.textContent().catch(() => null);
  if (t) seen.add(t.trim());
  if (seen.has("Completed")) break;
  await page.waitForTimeout(100);
}
for (const stage of ["Queued", "Preparing", "Generating", "Finalizing"]) {
  check(`pipeline reaches "${stage}"`, seen.has(stage));
}
// On completion the canvas swaps the progress view (which carries role=status)
// for the result view, so "Completed" is never rendered as a stage label —
// the result appearing IS the completion signal. The job strip keeps the word.
await page.waitForTimeout(500);
check(
  "completion swaps progress for the result view",
  await page.locator("main").getByRole("button", { name: "Download" }).first().isVisible(),
);
check(
  'job strip marks the job "Completed"',
  (await page.locator('[role="listitem"]', { hasText: "Completed" }).count()) >= 1,
);

check(
  "simulation disclosure visible during/after generation",
  await page.getByText("Simulated preview — AI generation is not connected").first().isVisible(),
);

// ── Step 6: capability-driven result actions (video) ──────────────────────
for (const label of ["Download", "Extend", "Reuse Settings", "Variation"]) {
  check(`video result shows "${label}"`, await page.getByRole("link", { name: label }).or(page.getByRole("button", { name: label })).first().isVisible());
}

// ── Step 5: a second concurrent job + job strip switching ─────────────────
await promptBox.fill("A second generation running alongside the first");
await generateBtn.click();
await page.waitForTimeout(400);
const jobCards = page.locator('[role="list"][aria-label="Generation jobs"] [role="listitem"]');
check("job strip shows 2 jobs", (await jobCards.count()) === 2, `${await jobCards.count()} found`);
check(
  "Generate label becomes 'Generate Another' while running",
  (await generateBtn.textContent())?.includes("Generate Another"),
);
check(
  "active-jobs indicator appears in the shell",
  await page.locator('[role="status"]', { hasText: "generation" }).first().isVisible(),
);
// aria-pressed lives on the button INSIDE the listitem wrapper: `listitem`
// does not support aria-pressed, so the two roles are deliberately split.
await jobCards.nth(1).locator("button").click();
check(
  "clicking a job in the strip selects it",
  (await jobCards.nth(1).locator("button").getAttribute("aria-pressed")) === "true",
);

// ── Settings preservation on workflow switch ──────────────────────────────
await page.goto(`${BASE}/app/create/text-to-video`, { waitUntil: "networkidle" });
await page.getByRole("button", { name: "15s", exact: true }).click();
check(
  "Text to Video accepts 15s",
  (await page.getByRole("button", { name: "15s", exact: true }).getAttribute("aria-pressed")) === "true",
);
await page.getByRole("link", { name: "Image to Video" }).first().click();
await page.waitForURL("**/create/image-to-video");
await page.waitForTimeout(300);
const has15 = await page.getByRole("button", { name: "15s", exact: true }).count();
const fiveNow = await page
  .getByRole("button", { name: "5s", exact: true })
  .getAttribute("aria-pressed");
check("Image to Video does not offer 15s", has15 === 0);
check("unsupported 15s falls back to 5s (no crash, no stale value)", fiveNow === "true");

// ── Music: audio workflow hides video-only controls ───────────────────────
await page.goto(`${BASE}/app/create/music`, { waitUntil: "networkidle" });
check("Music hides ASPECT RATIO section", (await page.getByText("Aspect ratio", { exact: true }).count()) === 0);
check("Music hides QUALITY section", (await page.getByText("Quality", { exact: true }).count()) === 0);
check("Music offers 30s/60s/120s", (await page.getByRole("button", { name: "120s", exact: true }).count()) === 1);
check(
  "Music empty state says 'track'",
  await page.getByText("Your generated track will appear here.").isVisible(),
);

await page.locator("#zx-prompt").fill("Synthwave track, 120bpm");
const musicGenerate = page
  .locator('aside[aria-label="Generation settings"] button', { hasText: "Generate" })
  .last();
await musicGenerate.click();
await page.waitForTimeout(7200);
// Scope to <main>: the sidebar nav and the workflow selector both contain an
// "Extend Video" control, which a loose name match would wrongly count.
const resultArea = page.locator("main");
check(
  "Music result shows Download",
  await resultArea.getByRole("button", { name: "Download" }).first().isVisible(),
);
check(
  "Music result has NO Extend (capability-driven)",
  (await resultArea.getByRole("link", { name: "Extend", exact: true }).count()) === 0,
);
check(
  "Music result still shows Variation + Reuse Settings",
  (await resultArea.getByRole("button", { name: "Variation" }).count()) === 1 &&
    (await resultArea.getByRole("button", { name: "Reuse Settings" }).count()) === 1,
);

// ── Accessibility spot checks ─────────────────────────────────────────────
await page.goto(`${BASE}/app/create/text-to-video`, { waitUntil: "networkidle" });
const a11y = await page.evaluate(() => {
  const out = {};
  const panel = document.querySelector('aside[aria-label="Generation settings"]');
  out.panelRoleDesktop = panel?.getAttribute("role");
  out.pressedCount = document.querySelectorAll("[aria-pressed]").length;
  out.currentCount = document.querySelectorAll('[aria-current="page"]').length;
  out.h1 = document.querySelectorAll("h1").length;
  out.imgsNoAlt = [...document.querySelectorAll("img")].filter((i) => !i.alt).length;
  out.labelledPrompt = !!document.querySelector('label[for="zx-prompt"]');
  return out;
});
check("desktop panel is NOT a dialog", a11y.panelRoleDesktop === null);
check("option chips expose aria-pressed", a11y.pressedCount > 5, `${a11y.pressedCount}`);
check("active nav marked aria-current", a11y.currentCount >= 1, `${a11y.currentCount}`);
check("exactly one <h1>", a11y.h1 === 1, `${a11y.h1}`);
check("prompt textarea has a <label for>", a11y.labelledPrompt);

// Escape closes the mobile settings sheet
const mobile = await browser.newPage({ viewport: { width: 390, height: 844 } });
await mobile.goto(`${BASE}/app/create/text-to-video`, { waitUntil: "networkidle" });
await mobile.getByRole("button", { name: "Settings" }).click();
await mobile.waitForTimeout(300);
const sheetOpen = await mobile.locator('aside[aria-label="Generation settings"]').isVisible();
const bodyLocked = await mobile.evaluate(() => document.body.style.overflow);
await mobile.keyboard.press("Escape");
await mobile.waitForTimeout(300);
const sheetClosed = !(await mobile.locator('aside[aria-label="Generation settings"]').isVisible());
const bodyFree = await mobile.evaluate(() => document.body.style.overflow);
check("mobile settings sheet opens", sheetOpen);
check("body scroll locks while sheet open", bodyLocked === "hidden", bodyLocked);
check("Escape closes the sheet", sheetClosed);
check("body scroll restored after close", bodyFree !== "hidden", bodyFree || "(empty)");

check("no uncaught page errors", errors.length === 0, errors.slice(0, 2).join(" | "));

await browser.close();

// ── Report ────────────────────────────────────────────────────────────────
const failed = results.filter((r) => !r.pass);
for (const r of results) {
  console.log(`${r.pass ? "PASS" : "FAIL"}  ${r.name}${r.detail ? "  (" + r.detail + ")" : ""}`);
}
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
