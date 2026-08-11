import { featureFlags } from "@/config/feature-flags";
import { cn } from "@/lib/cn";

/**
 * Build-state disclosure.
 *
 * The PRE-M1 version said AI generation was simulated. That is no longer true
 * and the wording has been corrected rather than carried forward: a generation
 * now creates a real job, a real worker executes it, and progress streams from
 * the backend. The `DemoSimulationNote` that sat under the progress bar has
 * been deleted outright, because leaving it would actively misinform.
 *
 * What IS still unfinished is the account layer — sign-in, plans and billing —
 * so that is what these two say, and nothing more.
 *
 * Both disappear when `featureFlags.previewBuild` is false.
 */

export function DemoBadge({
  compact = false,
  className,
}: {
  compact?: boolean;
  className?: string;
}) {
  if (!featureFlags.previewBuild) return null;

  return (
    <span
      className={cn(
        "text-zx-text-muted inline-flex items-center gap-[6px] text-[10.5px] font-bold tracking-[0.06em] uppercase",
        className,
      )}
    >
      <span
        aria-hidden="true"
        className="bg-zx-text-muted/60 inline-block h-[5px] w-[5px] rounded-full"
      />
      {compact ? "Preview" : "Preview build"}
    </span>
  );
}

export function DemoFooterNote({ className }: { className?: string }) {
  if (!featureFlags.previewBuild) return null;

  return (
    <p className={cn("text-zx-text-muted m-0 text-[12px] leading-[1.5]", className)}>
      Preview build — accounts and billing are not connected yet.
    </p>
  );
}
