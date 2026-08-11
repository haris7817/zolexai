"use client";

import { isRunning, type GenerationJob } from "@zolexai/workflow-contracts";
import { StatusDot, type StatusTone } from "@/components/ui/Feedback";
import { useRecentGenerations } from "@/features/generation/queries";
import { cn } from "@/lib/cn";

/**
 * Horizontal strip of recent generations.
 *
 * Reads the API rather than an in-tab store, which is a real improvement over
 * the PRE-M1 behaviour: jobs survive a page refresh, appear in a second tab,
 * and keep running while the user browses other screens — because they live in
 * PostgreSQL, not in this component's memory.
 */
export function JobStrip({
  selectedJobId,
  onSelect,
}: {
  selectedJobId: string | null;
  onSelect: (jobId: string) => void;
}) {
  const { jobs } = useRecentGenerations();

  if (jobs.length === 0) return null;

  return (
    <div
      role="list"
      aria-label="Recent generations"
      className="mt-[14px] flex gap-[10px] overflow-x-auto pb-[2px]"
    >
      {jobs.map((job) => (
        <JobCard
          key={job.id}
          job={job}
          selected={job.id === selectedJobId}
          onSelect={() => onSelect(job.id)}
        />
      ))}
    </div>
  );
}

export function statusTone(status: GenerationJob["status"]): StatusTone {
  if (status === "completed") return "success";
  if (status === "failed") return "error";
  if (status === "cancelled") return "muted";
  return "running";
}

function JobCard({
  job,
  selected,
  onSelect,
}: {
  job: GenerationJob;
  selected: boolean;
  onSelect: () => void;
}) {
  const running = isRunning(job.status);
  const thumb = job.outputs.find((output) => output.is_primary)?.url ?? null;
  const duration = String(job.parameters?.duration ?? "");

  return (
    // The source design put role="listitem" and aria-pressed on the same
    // button, which is invalid — `listitem` does not support aria-pressed.
    // Splitting them keeps both the list semantics and the toggle state.
    <div role="listitem" className="shrink-0">
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={selected}
        className={cn(
          "rounded-zx-md hover:border-zx-border-active flex shrink-0 cursor-pointer items-center gap-[10px] border px-3 py-[9px] text-left transition-colors duration-150",
          selected
            ? "bg-zx-surface-elevated border-zx-border-active"
            : "bg-zx-surface border-zx-border",
        )}
      >
        <span
          aria-hidden="true"
          className="relative h-8 w-12 shrink-0 overflow-hidden rounded-md bg-[#161A12] bg-cover bg-center"
          style={thumb ? { backgroundImage: `url(${thumb})` } : undefined}
        >
          {running ? (
            <span
              className="bg-zx-accent absolute bottom-0 left-0 h-[3px] transition-[width] duration-400"
              style={{ width: `${job.progress}%` }}
            />
          ) : null}
        </span>

        <span className="block">
          <span className="text-zx-text block max-w-[150px] truncate text-[11.5px] font-bold">
            {job.prompt || job.workflow_name}
          </span>
          <span className="mt-[3px] flex items-center gap-[6px]">
            <StatusDot tone={statusTone(job.status)} />
            <span className="text-zx-text-muted text-[10.5px] font-bold">
              {job.stage_label} · {job.workflow_name}
              {duration ? ` · ${duration}` : ""}
            </span>
          </span>
        </span>
      </button>
    </div>
  );
}
