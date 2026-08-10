"use client";

import {
  useGenerationJobs,
  selectSelectedJob,
} from "@/features/generation/useGenerationJobs";
import { isRunning, type GenerationJob } from "@/features/generation/types";
import { StatusDot, type StatusTone } from "@/components/ui/Feedback";
import { cn } from "@/lib/cn";

/**
 * Horizontal strip of this session's generations.
 *
 * Guide §7 Step 5 asks the demo to show a second generation starting while a
 * result already exists, then switching between them — this is that surface.
 * Jobs keep running while the user navigates, because they live in the store
 * rather than in this component.
 */
export function JobStrip() {
  const jobs = useGenerationJobs((state) => state.jobs);
  const selectJob = useGenerationJobs((state) => state.selectJob);
  const selectedJob = useGenerationJobs(selectSelectedJob);

  if (jobs.length === 0) return null;

  return (
    <div
      role="list"
      aria-label="Generation jobs"
      className="mt-[14px] flex gap-[10px] overflow-x-auto pb-[2px]"
    >
      {jobs.map((job) => (
        <JobCard
          key={job.id}
          job={job}
          selected={job.id === selectedJob?.id}
          onSelect={() => selectJob(job.id)}
        />
      ))}
    </div>
  );
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
  const tone: StatusTone =
    job.status === "Completed"
      ? "success"
      : job.status === "Failed"
        ? "error"
        : "running";

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
          className="relative h-8 w-12 shrink-0 overflow-hidden rounded-md"
          style={{ background: job.thumb }}
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
            {job.prompt}
          </span>
          <span className="mt-[3px] flex items-center gap-[6px]">
            <StatusDot tone={tone} />
            <span className="text-zx-text-muted text-[10.5px] font-bold">
              {job.status} · {job.workflowName} · {job.parameters.duration}
            </span>
          </span>
        </span>
      </button>
    </div>
  );
}
