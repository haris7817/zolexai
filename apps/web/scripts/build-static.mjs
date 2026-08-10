/**
 * Cross-platform static export.
 *
 * `NEXT_OUTPUT=export next build` is bash syntax; npm scripts run through
 * cmd.exe on Windows, where that fails. This sets the variable in-process and
 * spawns the build, so it works the same on Windows, macOS and Linux without
 * pulling in cross-env.
 *
 * Produces ./out — a plain static folder for drag-and-drop hosting.
 * Delete this at M1: once the frontend talks to FastAPI and SSE, the app is no
 * longer statically exportable.
 */
import { spawn } from "node:child_process";

const child = spawn("npx", ["next", "build"], {
  stdio: "inherit",
  shell: true,
  env: { ...process.env, NEXT_OUTPUT: "export" },
});

child.on("exit", (code) => process.exit(code ?? 1));
