import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const BASE = process.env.QA_BASE || "http://localhost:3000";
const OUT = process.argv[2] || "./shots";

const WIDTHS = [
  { w: 1440, h: 900, mode: "desktop" },
  { w: 1366, h: 850, mode: "laptop" },
  { w: 1024, h: 800, mode: "laptop" },
  { w: 768, h: 900, mode: "tablet" },
  { w: 430, h: 900, mode: "mobile" },
  { w: 390, h: 844, mode: "mobile" },
];

const ROUTES = [
  ["/", "landing"],
  ["/login", "login"],
  ["/register", "register"],
  ["/app", "dashboard"],
  ["/app/create/text-to-video", "workspace-t2v"],
  ["/app/create/music", "workspace-music"],
  ["/app/tools", "tools"],
  ["/app/generations", "generations"],
  ["/app/generations/gen_2479", "generation-detail"],
  ["/app/media", "media"],
  ["/app/subscription", "subscription"],
  ["/app/settings", "settings"],
];

fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const rows = [];
const problems = [];

for (const { w, h, mode } of WIDTHS) {
  const ctx = await browser.newContext({
    viewport: { width: w, height: h },
    deviceScaleFactor: 1,
    hasTouch: w < 768,
  });
  const page = await ctx.newPage();

  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text().slice(0, 200));
  });
  page.on("pageerror", (e) => consoleErrors.push("PAGEERROR " + String(e).slice(0, 200)));

  for (const [route, name] of ROUTES) {
    await page.goto(BASE + route, { waitUntil: "networkidle" });
    await page.waitForTimeout(250);

    const metrics = await page.evaluate(() => {
      const de = document.documentElement;
      const overflowX = de.scrollWidth - de.clientWidth;

      // Widest element that actually exceeds the viewport
      let worst = null;
      if (overflowX > 1) {
        for (const el of document.querySelectorAll("*")) {
          const r = el.getBoundingClientRect();
          if (r.right > de.clientWidth + 1 && r.width > 0) {
            const over = Math.round(r.right - de.clientWidth);
            if (!worst || over > worst.over) {
              worst = {
                over,
                tag: el.tagName.toLowerCase(),
                cls: (el.className || "").toString().slice(0, 70),
              };
            }
          }
        }
      }

      const sidebar = document.querySelector('[data-qa="sidebar"]');
      const panel = document.querySelector('aside[aria-label="Generation settings"]');
      const panelCS = panel ? getComputedStyle(panel) : null;

      // Touch targets: interactive elements smaller than 44px in either axis
      let smallTargets = 0;
      for (const el of document.querySelectorAll("button, a, input, [role=tab]")) {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        if (r.height < 28 || r.width < 20) smallTargets++;
      }

      return {
        overflowX,
        worst,
        sidebarW: sidebar ? Math.round(sidebar.getBoundingClientRect().width) : null,
        panelDisplay: panelCS ? panelCS.display : null,
        panelPosition: panelCS ? panelCS.position : null,
        panelW: panel ? Math.round(panel.getBoundingClientRect().width) : null,
        panelRole: panel ? panel.getAttribute("role") : null,
        smallTargets,
        title: document.title,
      };
    });

    rows.push({ w, mode, route, ...metrics });
    if (metrics.overflowX > 1) {
      problems.push(
        `OVERFLOW ${w}px ${route} +${metrics.overflowX}px via <${metrics.worst?.tag}> ${metrics.worst?.cls}`,
      );
    }

    await page.screenshot({
      path: path.join(OUT, `${name}-${w}.png`),
      fullPage: false,
    });
  }

  if (consoleErrors.length) {
    problems.push(`CONSOLE @${w}: ${[...new Set(consoleErrors)].slice(0, 4).join(" | ")}`);
  }

  await ctx.close();
}

await browser.close();

// ── Report ────────────────────────────────────────────────────────────────
console.log("\n=== HORIZONTAL OVERFLOW (must be 0 everywhere) ===");
const byWidth = {};
for (const r of rows) {
  byWidth[r.w] = (byWidth[r.w] || 0) + (r.overflowX > 1 ? 1 : 0);
}
for (const [w, count] of Object.entries(byWidth)) {
  console.log(`  ${w}px : ${count === 0 ? "CLEAN" : count + " route(s) overflow"}`);
}

console.log("\n=== SHELL MODE (sidebar width / panel) ===");
for (const w of WIDTHS.map((x) => x.w)) {
  const ws = rows.find((r) => r.w === w && r.route === "/app/create/text-to-video");
  const dash = rows.find((r) => r.w === w && r.route === "/app");
  console.log(
    `  ${String(w).padStart(4)}px  sidebar=${String(dash?.sidebarW).padStart(4)}  panel=${ws?.panelDisplay}/${ws?.panelPosition} w=${ws?.panelW} role=${ws?.panelRole}`,
  );
}

console.log("\n=== PROBLEMS ===");
console.log(problems.length ? problems.join("\n") : "  none");

fs.writeFileSync(path.join(OUT, "metrics.json"), JSON.stringify(rows, null, 2));
console.log(`\nScreenshots + metrics.json -> ${OUT}`);
