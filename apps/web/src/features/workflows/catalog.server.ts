import "server-only";

import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { parse as parseYaml } from "yaml";
import { workflowSchema, type Workflow } from "@zolexai/workflow-contracts";

/**
 * ===========================================================================
 * Build-time workflow catalogue — read straight from the YAML definitions
 * ===========================================================================
 *
 * The API is the runtime source for the *application*. This module exists for
 * everything that must not depend on the API being reachable:
 *
 *   · the landing page's tool grid — a public marketing page whose content
 *     should not vanish during an API deploy or outage
 *   · the app shell's navigation — rendered on the server before any query runs,
 *     so the sidebar is never a row of skeletons on first paint
 *
 * There is still only ONE source of truth. Python validates these files at API
 * startup and TypeScript parses the same files here; two independent readers of
 * one declarative file, which is loose coupling, not duplication. A test
 * asserts both readings are identical (`tests/catalog-parity.spec.ts`), so
 * drift fails the build rather than reaching a client.
 *
 * ## The private block
 *
 * A definition's `execution:` block carries runtime and (from M2) model and
 * provider detail. `toPublicWorkflow` below is an explicit ALLOWLIST — never a
 * `delete raw.execution` — because a denylist silently ships the next private
 * field somebody adds. Anything this function does not name cannot reach the
 * browser, and importing this module from a client component is a build error
 * (`server-only`).
 */

const DEFINITIONS_DIR =
  process.env.WORKFLOW_DEFINITIONS_DIR ??
  path.join(process.cwd(), "..", "..", "workflow-definitions");

interface RawInput {
  role: string;
  kind: string;
  required?: boolean;
  label: string;
  drop_hint: string;
  help?: string;
  accept?: string[];
  max_size_mb?: number;
}

interface RawWorkflow {
  id: string;
  version?: string;
  name: string;
  category: string;
  output_type: string;
  description: string;
  short_description?: string;
  marketing_description?: string;
  prompt?: { required?: boolean; placeholder?: string; max_length?: number };
  inputs?: RawInput[];
  supported_durations: string[];
  supported_aspect_ratios?: string[];
  supported_quality_levels?: string[];
  settings?: Record<string, boolean>;
  capabilities?: Record<string, boolean>;
  ui: { icon: string; thumb: string };
  // `execution` is intentionally absent from this interface as well as from the
  // projection: it must not be reachable from typed code on this side.
}

/** Product display order — the same sequence the API serves. */
const DISPLAY_ORDER = [
  "text-to-video",
  "image-to-video",
  "video-to-video",
  "extend-video",
  "music",
  "music-video",
];

function toPublicWorkflow(raw: RawWorkflow): Workflow {
  return workflowSchema.parse({
    id: raw.id,
    version: raw.version ?? "1",
    name: raw.name,
    category: raw.category,
    output_type: raw.output_type,
    description: raw.description,
    short_description: raw.short_description ?? "",
    marketing_description: raw.marketing_description ?? "",
    prompt: {
      required: raw.prompt?.required ?? true,
      placeholder: raw.prompt?.placeholder ?? "",
      max_length: raw.prompt?.max_length ?? 2000,
    },
    inputs: (raw.inputs ?? []).map((input) => ({
      role: input.role,
      kind: input.kind,
      required: input.required ?? true,
      label: input.label,
      drop_hint: input.drop_hint,
      help: input.help ?? "",
      accept: input.accept ?? [],
      max_size_mb: input.max_size_mb ?? 512,
    })),
    supported_durations: raw.supported_durations,
    supported_aspect_ratios: raw.supported_aspect_ratios ?? [],
    supported_quality_levels: raw.supported_quality_levels ?? [],
    settings: {
      quality: raw.settings?.quality ?? false,
      motion_strength: raw.settings?.motion_strength ?? false,
      prompt_adherence: raw.settings?.prompt_adherence ?? false,
      seed: raw.settings?.seed ?? false,
    },
    capabilities: {
      download: raw.capabilities?.download ?? true,
      extend: raw.capabilities?.extend ?? false,
      reuse_settings: raw.capabilities?.reuse_settings ?? true,
      variation: raw.capabilities?.variation ?? false,
    },
    ui: { icon: raw.ui.icon, thumb: raw.ui.thumb },
  });
}

// Parsed once per server process. The files are version-controlled and change
// only on deploy, so re-reading them per request would be pure overhead.
let cached: Workflow[] | null = null;

export async function loadWorkflowCatalog(): Promise<Workflow[]> {
  if (cached) return cached;

  const entries = await readdir(DEFINITIONS_DIR);
  const files = entries.filter((name) => name.endsWith(".yaml") && !name.startsWith("_"));

  const parsed = await Promise.all(
    files.map(async (file) => {
      const source = await readFile(path.join(DEFINITIONS_DIR, file), "utf8");
      const raw = parseYaml(source) as RawWorkflow;
      if (raw.id !== file.replace(/\.yaml$/, "")) {
        throw new Error(`${file}: workflow id '${raw.id}' does not match the filename`);
      }
      return toPublicWorkflow(raw);
    }),
  );

  parsed.sort((a, b) => {
    const ai = DISPLAY_ORDER.indexOf(a.id);
    const bi = DISPLAY_ORDER.indexOf(b.id);
    return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi);
  });

  cached = parsed;
  return parsed;
}

export async function loadWorkflow(id: string): Promise<Workflow | undefined> {
  return (await loadWorkflowCatalog()).find((workflow) => workflow.id === id);
}
