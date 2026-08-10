import Link from "next/link";
import { StatusPill, type StatusTone } from "@/components/ui/Feedback";
import { Icon } from "@/components/ui/Icon";
import type { MockGeneration } from "@/mocks/generations";

/**
 * A single generation in a grid — shared by the Creator Dashboard's "Recent
 * generations" and the Generations screen.
 */

export function statusTone(status: MockGeneration["status"]): StatusTone {
  if (status === "Completed") return "success";
  if (status === "Failed") return "error";
  return "running";
}

export function GenerationCard({ generation }: { generation: MockGeneration }) {
  const tone = statusTone(generation.status);
  const isAudio = generation.outputType === "audio";

  return (
    <Link
      href={`/app/generations/${generation.id}`}
      className="bg-zx-surface border-zx-border hover:border-zx-border-active block overflow-hidden rounded-[14px] border transition-colors duration-150"
    >
      <div
        className="relative flex items-center justify-center"
        style={{ background: generation.thumb, aspectRatio: "16 / 10" }}
      >
        <span className="absolute top-[10px] left-[10px]">
          <StatusPill tone={tone}>
            {generation.status.toUpperCase()}
          </StatusPill>
        </span>

        {generation.status === "Completed" ? (
          <span className="flex h-10 w-10 items-center justify-center rounded-full border border-white/25 bg-white/10 text-white backdrop-blur-[4px]">
            <Icon name={isAudio ? "audio" : "play"} size={16} />
          </span>
        ) : null}

        {generation.status === "Generating" &&
        typeof generation.progress === "number" ? (
          <span
            aria-hidden="true"
            className="bg-zx-accent absolute bottom-0 left-0 h-[3px]"
            style={{ width: `${generation.progress}%` }}
          />
        ) : null}

        <span className="text-zx-text absolute right-[10px] bottom-[10px] rounded-md bg-[rgba(13,12,19,0.7)] px-2 py-[3px] text-[10.5px] font-bold">
          {generation.duration}
        </span>
      </div>

      <div className="px-[14px] py-3">
        <div className="text-zx-text truncate text-[13px] font-bold">
          {generation.prompt}
        </div>
        <div className="text-zx-text-muted mt-[3px] text-[11.5px]">
          {generation.workflowName} · {generation.createdLabel}
        </div>
      </div>
    </Link>
  );
}
