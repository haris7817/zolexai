"use client";

import { useActiveGenerations } from "@/features/generation/queries";
import { cn } from "@/lib/cn";

/**
 * "N generations running" — the shell-level signal that work continues while
 * the user browses elsewhere.
 *
 * Reads the API, not this tab's memory, so it also counts a job started in
 * another tab or before a page refresh. Jobs live in PostgreSQL and keep
 * running whatever the browser does.
 *
 * Renders nothing when idle, so it never occupies space it hasn't earned.
 */
export function ActiveJobsIndicator({
  variant = "sidebar",
}: {
  variant?: "sidebar" | "header";
}) {
  const { jobs: activeJobs } = useActiveGenerations();
  if (activeJobs.length === 0) return null;

  const count = activeJobs.length;
  const label = count === 1 ? "1 generation running" : `${count} generations running`;
  const progress = activeJobs[0]?.progress ?? 0;

  if (variant === "header") {
    return (
      <span
        role="status"
        aria-label={label}
        className="border-zx-border-active bg-zx-primary/12 text-zx-primary-light inline-flex items-center gap-[7px] rounded-full border px-[12px] py-[5px] text-[11.5px] font-extrabold whitespace-nowrap"
      >
        <span
          aria-hidden="true"
          className="bg-zx-accent animate-zx-pulse h-[6px] w-[6px] rounded-full"
        />
        {count}
      </span>
    );
  }

  return (
    <div
      role="status"
      className="border-zx-border-active bg-zx-primary/10 rounded-zx-md mt-3 border p-[10px] laptop:p-3"
    >
      <div
        className={cn(
          "text-zx-primary-light flex items-center gap-2 text-[11.5px] font-extrabold",
          "justify-center laptop:justify-start",
        )}
      >
        <span
          aria-hidden="true"
          className="bg-zx-accent animate-zx-pulse h-[7px] w-[7px] shrink-0 rounded-full"
        />
        <span className="hidden laptop:inline">{label}</span>
      </div>

      <div className="mt-[9px] hidden h-1 overflow-hidden rounded-sm bg-white/8 laptop:block">
        <div
          className="h-full rounded-sm bg-[image:var(--zx-gradient-primary)] transition-[width] duration-400"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
