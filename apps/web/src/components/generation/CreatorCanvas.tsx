"use client";

import type { GenerationJob, Workflow } from "@zolexai/workflow-contracts";
import { brand } from "@/config/brand";
import { Icon } from "@/components/ui/Icon";
import { MediaPreview } from "@/components/media/MediaPreview";
import { primaryOutput } from "@/services/generations";
import { cn } from "@/lib/cn";
import { ResultActions } from "./ResultActions";

/** Audio has no picture; this is the height of its waveform panel. */
const AUDIO_SHAPE = { css: "16 / 6", ratio: 16 / 6 } as const;
const DEFAULT_SHAPE = { css: "16 / 9", ratio: 16 / 9 } as const;

/**
 * The shape to draw a finished result in.
 *
 * Measured dimensions first, because they are what the file actually is. The
 * requested aspect ratio is the fallback for older jobs whose assets predate
 * the API reporting width and height, and 16:9 is the last resort.
 *
 * Getting this from the request alone would be wrong in a way that matters:
 * Video to Video and Music Video take their shape from an uploaded file and
 * carry no `aspect_ratio` parameter at all.
 */
function previewShape(job: GenerationJob): { css: string; ratio: number } {
  const output = primaryOutput(job);
  if (output?.kind === "audio") return AUDIO_SHAPE;

  if (output?.width && output?.height) {
    return { css: `${output.width} / ${output.height}`, ratio: output.width / output.height };
  }

  const requested = String(job.parameters?.aspect_ratio ?? "");
  const [w, h] = requested.split(":").map(Number);
  if (w > 0 && h > 0) return { css: `${w} / ${h}`, ratio: w / h };

  return DEFAULT_SHAPE;
}

/**
 * How long the delivered file is — not how long was requested.
 *
 * On an extension those differ and the difference is the whole point: asking
 * to extend by 5s produces a 14-second video, and labelling that "5s" is what
 * the client saw. Falls back to the requested value only when the asset has no
 * measured duration.
 */
function resultDuration(job: GenerationJob): string {
  const seconds = primaryOutput(job)?.duration_seconds;
  if (seconds && seconds > 0) {
    const whole = Math.round(seconds);
    if (whole < 60) return `${whole}s`;
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  }
  return String(job.parameters?.duration ?? "");
}

/**
 * The centre of the creator workspace. Four states:
 *
 *   empty     nothing selected yet
 *   progress  a real job running, driven by SSE
 *   result    the completed output from object storage
 *   failed    a safe, human explanation of what went wrong
 *
 * The progress numbers here are reported by the worker over Server-Sent Events;
 * nothing in this file simulates anything. (The PRE-M1 timer that used to drive
 * it has been deleted.)
 */

/**
 * Shown only when this tool has never produced anything.
 *
 * There is no longer a "finished, look below" variant: a completed result stays
 * in the canvas so it can be played there (client revision, 14 Aug 2026), so the
 * only way to see this panel is an empty tool.
 */
export function EmptyGenerationState({ workflow }: { workflow: Workflow }) {
  const noun = workflow.output_type === "audio" ? "track" : "video";

  return (
    <div className="max-w-[380px] p-8 text-center">
      <div
        aria-hidden="true"
        className="mx-auto mb-[22px] flex aspect-video w-[120px] items-center justify-center rounded-[10px] border-[1.5px] border-dashed border-white/14"
      >
        <span className="text-zx-on-primary flex h-[30px] w-[30px] items-center justify-center rounded-[9px] bg-[image:var(--zx-gradient-primary)] text-[15px] font-extrabold opacity-55">
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

export function GenerationProgress({
  job,
  onCancel,
  cancelling,
}: {
  job: GenerationJob;
  onCancel: () => void;
  cancelling: boolean;
}) {
  return (
    <div className="animate-zx-fade-up w-[max(220px,min(520px,100%-56px,calc((100vh-380px)*1.78)))] text-center">
      <div className="rounded-zx-md border-zx-border animate-zx-shimmer relative mb-[22px] aspect-video border bg-[linear-gradient(110deg,var(--zx-surface)_30%,#1E2418_50%,var(--zx-surface)_70%)] bg-[length:200%_100%]">
        {/* The brand mark, pulsing in the centre while the job runs (client
            ask, 17 Aug 2026). Decorative — progress is announced by the
            aria-live stage label below, so this stays aria-hidden. */}
        <span
          aria-hidden="true"
          className="absolute inset-0 flex items-center justify-center"
        >
          <span className="text-zx-on-primary animate-pulse flex h-14 w-14 items-center justify-center rounded-[16px] bg-[image:var(--zx-gradient-primary)] text-[26px] font-extrabold shadow-[0_0_40px_rgba(163,230,53,0.35)]">
            {brand.shortName}
          </span>
        </span>
      </div>

      {/* aria-live: a screen reader announces each stage change without the
          user having to go looking for it. */}
      <div role="status" className="text-zx-text mb-[5px] text-[15.5px] font-extrabold">
        {job.stage_label}
      </div>
      <div className="text-zx-text-secondary mb-4 min-h-[18px] text-[12.5px]">
        {job.hint}
      </div>

      <div aria-hidden="true" className="h-[5px] overflow-hidden rounded-[3px] bg-white/7">
        <div
          className="h-full rounded-[3px] bg-[image:var(--zx-gradient-primary)] transition-[width] duration-400 ease-out"
          style={{ width: `${job.progress}%` }}
        />
      </div>

      <button
        type="button"
        onClick={onCancel}
        disabled={cancelling}
        className="text-zx-text-muted hover:text-zx-text-secondary mt-4 cursor-pointer text-[12px] font-bold transition-colors duration-150 disabled:cursor-default disabled:opacity-50"
      >
        {cancelling ? "Cancelling…" : "Cancel generation"}
      </button>
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
  workflow: Workflow;
  onReuseSettings: () => void;
  onVariation: () => void;
}) {
  const output = primaryOutput(job);
  const shape = previewShape(job);
  const duration = resultDuration(job);

  return (
    /* Width is bounded by the viewport height TIMES THE RESULT'S OWN RATIO.
       The 1.78 that used to be hard-coded here is 16:9, so a portrait video
       was sized as though it were landscape and overflowed its space — one
       half of the client's "you can not see it on the box" report. */
    <div
      className="animate-zx-fade-up"
      style={{
        width: `max(220px, min(720px, 100% - 48px, calc((100vh - 360px) * ${shape.ratio})))`,
      }}
    >
      <MediaPreview
        url={output?.url ?? null}
        kind={output?.kind ?? "image"}
        aspectRatio={shape.css}
        fallbackGradient={workflow.ui.thumb}
        className="rounded-zx-md border-zx-border border shadow-[0_24px_70px_rgba(0,0,0,0.45)]"
      />

      {duration ? (
        <div className="text-zx-text-muted mt-2 text-right text-[11px] font-bold">
          {duration}
        </div>
      ) : null}

      <ResultActions
        job={job}
        workflow={workflow}
        onReuseSettings={onReuseSettings}
        onVariation={onVariation}
      />
    </div>
  );
}

/**
 * Failure state.
 *
 * Shows the API's customer-safe message and nothing else. Worker internals,
 * stack traces and model names are sanitized server-side and never reach here
 * (architecture rule #10, directive §23) — so there is nothing to filter in
 * this component, by design.
 */
export function GenerationFailed({
  job,
  onRetry,
}: {
  job: GenerationJob;
  onRetry: () => void;
}) {
  const cancelled = job.status === "cancelled";

  return (
    <div className="animate-zx-fade-up max-w-[420px] p-8 text-center">
      <div
        className={cn(
          "mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-[14px] border",
          cancelled
            ? "border-zx-border bg-zx-surface text-zx-text-muted"
            : "border-zx-error/40 bg-zx-error/8 text-zx-error",
        )}
      >
        <Icon name={cancelled ? "close" : "alert"} size={20} />
      </div>

      <div className="text-zx-text mb-[7px] text-[15.5px] font-extrabold">
        {cancelled ? "Generation cancelled" : "Generation failed"}
      </div>
      <p className="text-zx-text-secondary mb-5 text-[13px] leading-[1.55]">
        {job.error?.message ??
          (cancelled
            ? "You stopped this generation before it finished."
            : "This generation could not be completed. Please try again.")}
      </p>

      <button
        type="button"
        onClick={onRetry}
        className="text-zx-primary-light hover:text-zx-text cursor-pointer text-[12.5px] font-bold transition-colors duration-150"
      >
        Reuse these settings
      </button>
    </div>
  );
}
