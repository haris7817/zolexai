"use client";

import { brand } from "@/config/brand";
import { DemoSimulationNote } from "@/components/ui/DemoDisclosure";
import { ResultActions } from "./ResultActions";
import type { GenerationJob } from "@/features/generation/types";
import type { WorkflowDefinition } from "@/features/workflows/types";

/**
 * The centre of the creator workspace. Three states, exactly as approved:
 *
 *   empty     nothing generated yet
 *   progress  simulated pipeline running
 *   result    completed placeholder output + capability-driven actions
 */

export function EmptyGenerationState({
  workflow,
}: {
  workflow: WorkflowDefinition;
}) {
  const noun = workflow.outputType === "audio" ? "track" : "video";

  return (
    <div className="max-w-[380px] p-8 text-center">
      <div
        aria-hidden="true"
        className="mx-auto mb-[22px] flex aspect-video w-[120px] items-center justify-center rounded-[10px] border-[1.5px] border-dashed border-white/14"
      >
        <span className="flex h-[30px] w-[30px] items-center justify-center rounded-[9px] bg-[image:var(--zx-gradient-primary)] text-[15px] font-extrabold text-zx-on-primary opacity-55">
          {brand.shortName}
        </span>
      </div>
      <div className="text-zx-text mb-[7px] text-[16px] font-extrabold">
        Your generated {noun} will appear here.
      </div>
      <p className="text-zx-text-secondary text-[13.5px] leading-[1.55]">
        Describe your idea, choose your settings and hit Generate.
      </p>
    </div>
  );
}

export function GenerationProgress({ job }: { job: GenerationJob }) {
  return (
    <div className="animate-zx-fade-up w-[max(220px,min(520px,100%-56px,calc((100vh-380px)*1.78)))] text-center">
      <div className="rounded-zx-md border-zx-border animate-zx-shimmer mb-[22px] aspect-video border bg-[linear-gradient(110deg,var(--zx-surface)_30%,#1E2418_50%,var(--zx-surface)_70%)] bg-[length:200%_100%]" />

      <div
        role="status"
        className="text-zx-text mb-[5px] text-[15.5px] font-extrabold"
      >
        {job.status}
      </div>
      <div className="text-zx-text-secondary mb-4 text-[12.5px]">
        {job.hint}
      </div>

      <div
        aria-hidden="true"
        className="h-[5px] overflow-hidden rounded-[3px] bg-white/7"
      >
        <div
          className="h-full rounded-[3px] bg-[image:var(--zx-gradient-primary)] transition-[width] duration-400 ease-out"
          style={{ width: `${job.progress}%` }}
        />
      </div>

      {/* Sits exactly where a moving progress bar could be mistaken for real
          inference — the load-bearing disclosure (guide §7 Step 4). */}
      <DemoSimulationNote className="mt-4" />
    </div>
  );
}

export function GenerationResult({
  job,
  workflow,
  onReuseSettings,
  onVariation,
}: {
  job: GenerationJob;
  workflow: WorkflowDefinition;
  onReuseSettings: () => void;
  onVariation: () => void;
}) {
  const isAudio = workflow.outputType === "audio";

  return (
    <div className="animate-zx-fade-up w-[max(220px,min(720px,100%-48px,calc((100vh-360px)*1.78)))]">
      <div
        className="rounded-zx-md border-zx-border relative flex items-center justify-center overflow-hidden border shadow-[0_24px_70px_rgba(0,0,0,0.45)]"
        style={{
          background: job.thumb,
          aspectRatio: isAudio ? "16 / 6" : "16 / 9",
        }}
      >
        <div
          aria-hidden="true"
          className="absolute inset-0 bg-[radial-gradient(circle_at_65%_25%,rgba(198,242,36,0.13),transparent_60%)]"
        />

        {isAudio ? <AudioWaveform /> : null}

        <button
          type="button"
          aria-label={isAudio ? "Play track" : "Play video"}
          className="relative flex h-[60px] w-[60px] cursor-pointer items-center justify-center rounded-full border border-white/25 bg-white/10 backdrop-blur-[6px] transition-colors duration-150 hover:bg-white/18"
        >
          <span
            aria-hidden="true"
            className="ml-[4px] h-0 w-0 border-y-[10px] border-l-[16px] border-y-transparent border-l-white"
          />
        </button>

        <span className="text-zx-text absolute right-[14px] bottom-3 rounded-md bg-[rgba(10,10,11,0.7)] px-[9px] py-[4px] text-[11px] font-bold">
          {job.parameters.duration}
        </span>
      </div>

      <ResultActions
        workflow={workflow}
        onReuseSettings={onReuseSettings}
        onVariation={onVariation}
      />

      <DemoSimulationNote className="mt-4" />
    </div>
  );
}

/**
 * Audio results get a waveform rather than a video frame — architecture doc
 * §31 requires audio to be a first-class output type, not a video with the
 * picture missing.
 */
function AudioWaveform() {
  const bars = [
    18, 34, 26, 48, 62, 40, 72, 54, 88, 66, 44, 78, 58, 92, 70, 46, 84, 60, 38,
    52, 30, 64, 42, 24,
  ];
  return (
    <div
      aria-hidden="true"
      className="absolute inset-x-8 bottom-6 flex items-end justify-center gap-[3px] opacity-45"
    >
      {bars.map((height, index) => (
        <span
          key={index}
          className="w-[3px] rounded-full bg-white/70"
          style={{ height: `${height}%` }}
        />
      ))}
    </div>
  );
}
