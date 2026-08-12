import Link from "next/link";
import { isRunning, type GenerationJob } from "@zolexai/workflow-contracts";
import { StatusPill, type StatusTone } from "@/components/ui/Feedback";
import { Icon } from "@/components/ui/Icon";
import { primaryOutput } from "@/services/generations";
import { durationLabel } from "@/services/workflows";

/**
 * A single generation in a grid — shared by the Creator Dashboard's "Recent
 * generations" and the Generations screen.
 */

export function statusTone(status: GenerationJob["status"]): StatusTone {
  if (status === "completed") return "success";
  if (status === "failed") return "error";
  if (status === "cancelled") return "muted";
  return "running";
}

/**
 * Relative time, computed on the client only.
 *
 * `suppressHydrationWarning` because the server and the browser render this a
 * fraction of a second apart, and "2 minutes ago" vs "3 minutes ago" would
 * otherwise be a hydration mismatch on every history page.
 */
export function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.round(hours / 24);
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  const weeks = Math.round(days / 7);
  return `${weeks} week${weeks === 1 ? "" : "s"} ago`;
}

export function GenerationCard({ generation }: { generation: GenerationJob }) {
  const output = primaryOutput(generation);
  // Absent for automatic-duration workflows — the badge simply doesn't render.
  const duration = durationLabel(String(generation.parameters?.duration ?? ""));
  const running = isRunning(generation.status);

  return (
    <Link
      href={`/app/generations/${generation.id}`}
      className="bg-zx-surface border-zx-border hover:border-zx-border-active block overflow-hidden rounded-[14px] border transition-colors duration-150"
    >
      <div
        className="relative flex items-center justify-center bg-[#141A0C] bg-cover bg-center"
        style={{
          aspectRatio: "16 / 10",
          backgroundImage: output?.url ? `url(${output.url})` : undefined,
        }}
      >
        <span className="absolute top-[10px] left-[10px]">
          <StatusPill tone={statusTone(generation.status)}>
            {generation.stage_label.toUpperCase()}
          </StatusPill>
        </span>

        {generation.status === "completed" ? (
          <span className="flex h-10 w-10 items-center justify-center rounded-full border border-white/25 bg-white/10 text-white backdrop-blur-[4px]">
            <Icon name={output?.kind === "audio" ? "audio" : "play"} size={16} />
          </span>
        ) : null}

        {running ? (
          <span
            aria-hidden="true"
            className="bg-zx-accent absolute bottom-0 left-0 h-[3px] transition-[width] duration-400"
            style={{ width: `${generation.progress}%` }}
          />
        ) : null}

        {duration ? (
          <span className="text-zx-text absolute right-[10px] bottom-[10px] rounded-md bg-[rgba(10,10,11,0.7)] px-2 py-[3px] text-[10.5px] font-bold">
            {duration}
          </span>
        ) : null}
      </div>

      <div className="px-[14px] py-3">
        <div className="text-zx-text truncate text-[13px] font-bold">
          {generation.prompt || generation.workflow_name}
        </div>
        <div className="text-zx-text-muted mt-[3px] text-[11.5px]" suppressHydrationWarning>
          {generation.workflow_name} · {relativeTime(generation.created_at)}
        </div>
      </div>
    </Link>
  );
}
