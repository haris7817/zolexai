import type { NextConfig } from "next";
import fs from "node:fs";
import path from "node:path";

/**
 * Next.js configuration.
 *
 * ## The static export path is gone (was PRE-M1 only)
 *
 * The client demo was published as a static folder because every route was
 * mock-driven and could be prerendered. That is no longer true and cannot be
 * made true again: the app now talks to FastAPI, streams SSE, and reads
 * generation history that only exists at runtime — `/app/generations/[id]`
 * has no build-time set of ids to enumerate.
 *
 * The approved PRE-M1 demo remains live and untouched at its own URL; it is
 * served from a standalone copy of the old `out/` folder, so removing the
 * export path here does not affect it.
 */

const REPO_ROOT = path.join(__dirname, "../../");

/**
 * Loads the repo-root `.env`.
 *
 * The API and the worker both read `<repo>/.env` directly, so it is the one
 * place local configuration lives. Next only looks inside its own app
 * directory, which meant `NEXT_PUBLIC_API_URL` was silently absent from a build
 * run from the repo root — the browser then called its own origin for
 * `/api/v1/*` and every request 404'd, while server-rendered content kept
 * working and hid the problem.
 *
 * Values already present in the environment win, so CI and Docker builds
 * override this rather than fight it.
 */
function loadRepoEnv(): void {
  const file = path.join(REPO_ROOT, ".env");
  if (!fs.existsSync(file)) return;

  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    const match = /^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*)\s*$/.exec(line);
    if (!match) continue;
    const [, key, rawValue] = match;
    if (process.env[key] !== undefined) continue;
    process.env[key] = rawValue.trim().replace(/^["']|["']$/g, "");
  }
}

loadRepoEnv();

const nextConfig: NextConfig = {
  // Monorepo: pin tracing to the repo root so Next does not guess from the
  // nearest lockfile.
  outputFileTracingRoot: REPO_ROOT,

  // Shipped as TypeScript source (no build step), so Next must compile it.
  transpilePackages: ["@zolexai/workflow-contracts"],

  // Required for `docker compose --profile apps up`: bundles only the files the
  // server actually needs instead of the whole workspace node_modules.
  output: process.env.NEXT_STANDALONE === "1" ? "standalone" : undefined,

  env: {
    // Inlined into the client bundle at build time.
    //
    // Falls back to same-origin, which is the correct PRODUCTION default: the
    // API sits behind the same domain via a reverse proxy. It is deliberately
    // NOT a localhost default — that would turn a missing production variable
    // into a bundle that quietly points every user's browser at their own
    // machine.
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "",
  },

  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          // The app renders user-generated media but is never itself embedded.
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
    ];
  },
};

export default nextConfig;
