import type { NextConfig } from "next";
import path from "node:path";

/**
 * Every route in the PRE-M1 demo is Static or SSG — there is no backend and no
 * server rendering — so the app can also be emitted as a plain static folder.
 *
 *   npm run build          → normal Next build (Vercel, Netlify, any Node host)
 *   npm run build:static   → ./out, a drag-and-drop static bundle
 *
 * The static path exists so the client demo can be published without an account
 * anywhere that serves files. Delete it at M1: once the frontend talks to
 * FastAPI and SSE, static export is no longer valid.
 */
const isStaticExport = process.env.NEXT_OUTPUT === "export";

const nextConfig: NextConfig = {
  // Monorepo: pin tracing to the repo root so Next does not guess from
  // the nearest lockfile.
  outputFileTracingRoot: path.join(__dirname, "../../"),

  ...(isStaticExport
    ? {
        output: "export" as const,
        // Static hosts serve /path/index.html rather than /path
        trailingSlash: true,
        images: { unoptimized: true },
      }
    : {}),
};

export default nextConfig;
