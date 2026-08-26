/**
 * Catalogue parity — the guard against the one real risk of two readers.
 *
 * `workflow-definitions/*.yaml` is the single source of truth, but TWO
 * independent programs parse it: Python validates it at API startup, and
 * TypeScript reads it at render time for the landing page and app shell. That
 * is loose coupling by design (one declarative file, two readers), and its one
 * failure mode is drift — a field one side honours and the other quietly drops.
 *
 * This compares what the API actually serves against the YAML on disk, field by
 * field, and additionally asserts that nothing private escaped.
 *
 *   QA_API=http://localhost:8000 node scripts/qa-catalog-parity.mjs
 */
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { parse } from "yaml";

const API = process.env.QA_API || "http://localhost:8000";
const DIR = path.join(process.cwd(), "..", "..", "workflow-definitions");

/** Fields the API is expected to serve verbatim from the definition. */
const MIRRORED = [
  "id",
  "version",
  "name",
  "category",
  "output_type",
  "description",
  "short_description",
  "marketing_description",
  "duration_mode",
  "supported_durations",
  "supported_aspect_ratios",
  "supported_quality_levels",
  "supported_durations_by_quality",
];

/** Anything under here must never appear in a public response. */
const PRIVATE_KEYS = ["execution", "runtime", "output_content_type", "output_kind"];

const failures = [];
const fail = (message) => failures.push(message);

const names = (await readdir(DIR)).filter((n) => n.endsWith(".yaml") && !n.startsWith("_"));
const onDisk = new Map();

for (const name of names) {
  const raw = parse(await readFile(path.join(DIR, name), "utf8"));
  if (raw.id !== name.replace(/\.yaml$/, "")) {
    fail(`${name}: id "${raw.id}" does not match the filename`);
  }
  onDisk.set(raw.id, raw);
}

const response = await fetch(`${API}/api/v1/workflows`);
if (!response.ok) {
  console.error(`FAIL — GET /api/v1/workflows returned HTTP ${response.status}`);
  process.exit(1);
}
const bodyText = await response.text();
const served = new Map(JSON.parse(bodyText).workflows.map((w) => [w.id, w]));

// ── Same set of workflows ────────────────────────────────────────────────
for (const id of onDisk.keys()) {
  if (!served.has(id)) fail(`${id}: defined on disk but not served by the API`);
}
for (const id of served.keys()) {
  if (!onDisk.has(id)) fail(`${id}: served by the API but has no definition file`);
}

// ── Same values ──────────────────────────────────────────────────────────
for (const [id, disk] of onDisk) {
  const api = served.get(id);
  if (!api) continue;

  for (const field of MIRRORED) {
    const expected =
      disk[field] ??
      (Array.isArray(api[field])
        ? []
        : typeof api[field] === "object" && api[field] !== null
          ? {}
          : "");
    if (JSON.stringify(expected) !== JSON.stringify(api[field])) {
      fail(
        `${id}.${field}: yaml ${JSON.stringify(expected)} != api ${JSON.stringify(api[field])}`,
      );
    }
  }

  // Input roles, with their required-ness — this is what drives which
  // dropzones render and which are optional.
  const diskRoles = (disk.inputs ?? []).map((i) => `${i.role}:${i.required !== false}`);
  const apiRoles = api.inputs.map((i) => `${i.role}:${i.required}`);
  if (diskRoles.join(",") !== apiRoles.join(",")) {
    fail(`${id}.inputs: yaml [${diskRoles}] != api [${apiRoles}]`);
  }

  for (const key of [
    "quality",
    "motion_strength",
    "prompt_adherence",
    "seed",
    "lyrics",
    "prompt_modes",
    "sound",
  ]) {
    const expected = disk.settings?.[key] ?? false;
    if (expected !== api.settings[key]) {
      fail(`${id}.settings.${key}: yaml ${expected} != api ${api.settings[key]}`);
    }
  }

  for (const key of ["download", "extend", "reuse_settings", "variation"]) {
    const expected = disk.capabilities?.[key] ?? (key === "download" || key === "reuse_settings");
    if (expected !== api.capabilities[key]) {
      fail(`${id}.capabilities.${key}: yaml ${expected} != api ${api.capabilities[key]}`);
    }
  }

  if (disk.ui?.icon !== api.ui.icon || disk.ui?.thumb !== api.ui.thumb) {
    fail(`${id}.ui: yaml ${JSON.stringify(disk.ui)} != api ${JSON.stringify(api.ui)}`);
  }
}

// ── Nothing private escaped ──────────────────────────────────────────────
for (const key of PRIVATE_KEYS) {
  if (bodyText.toLowerCase().includes(key)) {
    fail(`private key "${key}" appears in the public catalogue response`);
  }
}
for (const name of ["ltx", "comfyui", "vast.ai", "pytorch", "cuda"]) {
  if (bodyText.toLowerCase().includes(name)) {
    fail(`internal name "${name}" is visible to clients`);
  }
}

if (failures.length === 0) {
  console.log(`PASS — ${onDisk.size} workflows: YAML and API agree, no private fields leaked`);
  process.exit(0);
}

console.log(`${failures.length} parity problem(s):\n`);
for (const line of failures) console.log(`  ${line}`);
process.exit(1);
