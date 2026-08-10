/**
 * Contrast QA — measures real rendered colours against WCAG AA.
 *
 * Exists because the lime accent theme makes some pairings that were safe on
 * violet unsafe (white on lime is ~1.3:1). This walks key surfaces and reports
 * anything below threshold rather than trusting the palette on paper.
 *
 *   QA_BASE=http://localhost:3220 node scripts/qa-contrast.mjs
 */
import { chromium } from "playwright";

const BASE = process.env.QA_BASE || "http://localhost:3000";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

const ROUTES = [
  "/",
  "/app",
  "/app/create/text-to-video",
  "/app/generations",
  "/app/subscription",
  "/login",
];

const results = [];

for (const route of ROUTES) {
  await page.goto(BASE + route, { waitUntil: "networkidle" });
  await page.waitForTimeout(400);

  const found = await page.evaluate(() => {
    const srgb = (c) => {
      c /= 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    };
    const lum = ([r, g, b]) =>
      0.2126 * srgb(r) + 0.7152 * srgb(g) + 0.0722 * srgb(b);
    const parse = (s) => (s.match(/\d+(\.\d+)?/g) || []).map(Number);

    // Colour stops inside a gradient, e.g. "linear-gradient(rgb(169,222,0), …)"
    const stopsOf = (bgImage) => {
      if (!bgImage || bgImage === "none") return [];
      const out = [];
      for (const m of bgImage.matchAll(/rgba?\(([^)]+)\)/g)) {
        const p = m[1].split(",").map((n) => parseFloat(n));
        // ignore near-transparent stops — they don't determine legibility
        if (p.length >= 3 && (p[3] === undefined || p[3] > 0.6)) {
          out.push(p.slice(0, 3));
        }
      }
      return out;
    };

    /**
     * Returns every plausible background behind an element.
     *
     * Critically this includes GRADIENT stops: a lime CTA has
     * background-color: transparent and its colour lives in background-image,
     * so walking up to the page background would compare dark-on-dark and
     * wrongly pass — or, as here, wrongly fail.
     */
    const backgroundsOf = (el) => {
      // Semi-transparent layers must be COMPOSITED, not skipped. A dark pill at
      // 0.7 alpha over a bright thumbnail is genuinely dark; ignoring it would
      // report a failure that doesn't exist on screen.
      const layers = [];
      let n = el;
      while (n && n !== document.documentElement) {
        const cs = getComputedStyle(n);
        const stops = stopsOf(cs.backgroundImage);
        if (stops.length) {
          layers.push({ colors: stops, alpha: 1 });
          break;
        }
        const p = parse(cs.backgroundColor);
        const a = p.length >= 4 ? p[3] : p.length === 3 ? 1 : 0;
        if (p.length >= 3 && a > 0.02) {
          layers.push({ colors: [p.slice(0, 3)], alpha: a });
          if (a > 0.98) break;
        }
        n = n.parentElement;
      }
      layers.push({ colors: [[10, 10, 11]], alpha: 1 });

      // Composite back-to-front for each combination of gradient stops
      const count = Math.max(...layers.map((l) => l.colors.length));
      const results = [];
      for (let i = 0; i < count; i++) {
        let acc = layers[layers.length - 1].colors[0];
        for (let j = layers.length - 2; j >= 0; j--) {
          const L = layers[j];
          const c = L.colors[Math.min(i, L.colors.length - 1)];
          acc = [0, 1, 2].map((k) => L.alpha * c[k] + (1 - L.alpha) * acc[k]);
        }
        results.push(acc);
      }
      return results;
    };

    const out = [];
    const seen = new Set();
    const nodes = document.querySelectorAll(
      "button, a, p, h1, h2, h3, span, div, label, li, td, dd",
    );
    for (const el of nodes) {
      const text = (el.textContent || "").trim();
      if (!text || text.length > 90) continue;
      if (el.children.length > 0) continue; // leaf text only
      const cs = getComputedStyle(el);
      if (cs.visibility === "hidden" || cs.display === "none") continue;
      const r = el.getBoundingClientRect();
      if (r.width < 4 || r.height < 4) continue;

      const px = parseFloat(cs.fontSize);
      const bold = parseInt(cs.fontWeight, 10) >= 700;
      const large = px >= 24 || (px >= 18.66 && bold);
      const need = large ? 3.0 : 4.5;

      let fg = parse(cs.color);
      const transparentText = fg.length >= 4 && fg[3] === 0;

      // bg-clip-text: the glyphs ARE the gradient, so the element's own
      // gradient stops are the FOREGROUND and the parent is the background.
      let candidateBgs;
      if (transparentText && /text/.test(cs.backgroundClip || cs.webkitBackgroundClip || "")) {
        const stops = stopsOf(cs.backgroundImage);
        if (!stops.length) continue;
        candidateBgs = backgroundsOf(el.parentElement || el);
        // worst-case: darkest gradient stop against the background
        fg = stops.reduce((a, b) => (lum(a) < lum(b) ? a : b));
      } else {
        if (fg.length < 3 || (fg[3] !== undefined && fg[3] < 0.5)) continue;
        candidateBgs = backgroundsOf(el);
      }

      // worst case across every possible background behind this text
      let worst = Infinity;
      let worstBg = candidateBgs[0];
      for (const bg of candidateBgs) {
        const L1 = lum(fg), L2 = lum(bg);
        const r2 = (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
        if (r2 < worst) { worst = r2; worstBg = bg; }
      }

      const key = fg.join(",") + "|" + worstBg.join(",") + "|" + Math.round(px);
      if (seen.has(key)) continue;
      seen.add(key);

      if (worst < need) {
        out.push({
          text: text.slice(0, 40),
          fg: `rgb(${fg.slice(0, 3).map(Math.round).join(",")})`,
          bg: `rgb(${worstBg.slice(0, 3).map(Math.round).join(",")})`,
          px,
          ratio: +worst.toFixed(2),
          need,
        });
      }
    }
    return out;
  });

  for (const f of found) results.push({ route, ...f });
}

await browser.close();

if (results.length === 0) {
  console.log("PASS — every sampled text/background pair meets WCAG AA");
} else {
  console.log(`${results.length} pairing(s) below AA:\n`);
  for (const r of results) {
    console.log(
      `  ${r.route}\n    "${r.text}"\n    ${r.fg} on ${r.bg} @${r.px}px → ${r.ratio}:1 (needs ${r.need}:1)`,
    );
  }
}
process.exit(results.length ? 1 : 0);
