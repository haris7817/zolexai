import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // Monorepo: pin tracing to the repo root so Next does not guess from
  // the nearest lockfile.
  outputFileTracingRoot: path.join(__dirname, "../../"),
};

export default nextConfig;
