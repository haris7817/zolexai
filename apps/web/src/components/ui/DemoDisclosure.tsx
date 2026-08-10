import { featureFlags } from "@/config/feature-flags";
import { cn } from "@/lib/cn";

/**
 * ===========================================================================
 * Demo disclosure — PREUI-15
 * ===========================================================================
 *
 * Purpose (guide §3, §4): the client must not mistake simulated progress and
 * placeholder output for connected AI.
 *
 * Deliberately subtle. Muted tokens only — no alarm colours, no modal, no
 * layout-shifting banner, and no repeated warning-style messaging. Three
 * placements, each doing a different job:
 *
 *   1. DemoBadge          persistent, low-key, always in the shell
 *   2. DemoSimulationNote at the exact point of risk — under the fake progress
 *                         bar and the placeholder result
 *   3. DemoFooterNote     one line on the public landing page
 *
 * All three disappear when `featureFlags.demoMode` is false, so M1 removes the
 * disclosure by flipping one boolean.
 */

/** Persistent, unobtrusive marker in the sidebar footer and mobile drawer. */
export function DemoBadge({
  className,
  compact = false,
}: {
  className?: string;
  compact?: boolean;
}) {
  if (!featureFlags.demoMode) return null;

  return (
    <span
      className={cn(
        "border-zx-border bg-zx-surface text-zx-text-muted inline-flex items-center gap-[6px] rounded-full border px-[10px] py-[4px] text-[10.5px] font-bold",
        className,
      )}
      title="AI generation is simulated for design review"
    >
      <span
        aria-hidden="true"
        className="bg-zx-text-muted h-[5px] w-[5px] rounded-full"
      />
      {compact ? "Preview" : "UI Preview"}
    </span>
  );
}

/**
 * The load-bearing one. Sits directly beneath the generation progress and the
 * generated result — where a client would otherwise read a moving progress bar
 * as real inference (guide §7 Step 4).
 */
export function DemoSimulationNote({ className }: { className?: string }) {
  if (!featureFlags.demoMode) return null;

  return (
    <p
      className={cn(
        "text-zx-text-muted text-center text-[11.5px] leading-[1.5]",
        className,
      )}
    >
      Simulated preview — AI generation is not connected in this demo.
    </p>
  );
}

/** Landing page footer line. */
export function DemoFooterNote({ className }: { className?: string }) {
  if (!featureFlags.demoMode) return null;

  return (
    <p className={cn("text-zx-text-muted text-[12.5px]", className)}>
      Interactive product preview for design review.
    </p>
  );
}
