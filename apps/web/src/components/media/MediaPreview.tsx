"use client";

import { useState } from "react";
import type { AssetKind } from "@zolexai/workflow-contracts";
import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

/**
 * Renders a stored asset by its ACTUAL kind, not by the workflow's declared
 * output type.
 *
 * That distinction matters right now: M1's mock runtime produces a placeholder
 * image for every workflow, including the video ones. Choosing the renderer
 * from `workflow.output_type` would mount a `<video>` around a PNG and show a
 * broken player. Choosing it from `asset.kind` renders what is genuinely there,
 * and keeps working unchanged when M2 starts producing real video and audio —
 * no component has to learn about the transition.
 */
export function MediaPreview({
  url,
  kind,
  poster,
  className,
  fallbackGradient,
  aspectRatio,
}: {
  url: string | null;
  kind: AssetKind;
  poster?: string;
  className?: string;
  /** Shown while there is no asset yet — queued, failed, or still uploading. */
  fallbackGradient?: string;
  aspectRatio?: string;
}) {
  const [failed, setFailed] = useState(false);

  const frame = cn("relative overflow-hidden", className);
  const style = { aspectRatio, background: fallbackGradient } as const;

  if (!url || failed) {
    return (
      <div className={frame} style={style} aria-hidden={!failed}>
        {failed ? (
          <div className="text-zx-text-muted absolute inset-0 flex flex-col items-center justify-center gap-2 text-[12px] font-bold">
            <Icon name="alert" size={18} />
            Preview unavailable
          </div>
        ) : null}
      </div>
    );
  }

  if (kind === "audio") {
    return (
      <div className={frame} style={style}>
        <AudioWaveform />
        <audio
          src={url}
          controls
          preload="metadata"
          onError={() => setFailed(true)}
          className="absolute inset-x-4 bottom-4 w-[calc(100%-2rem)]"
        />
      </div>
    );
  }

  if (kind === "video") {
    return (
      <div className={frame} style={style}>
        <video
          src={url}
          poster={poster}
          controls
          playsInline
          preload="metadata"
          onError={() => setFailed(true)}
          className="absolute inset-0 h-full w-full object-cover"
        />
      </div>
    );
  }

  return (
    <div className={frame} style={style}>
      {/* Plain <img>: these are presigned, short-lived, cross-origin storage
          URLs that the Next image optimiser cannot fetch or cache. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={url}
        alt=""
        loading="lazy"
        onError={() => setFailed(true)}
        className="absolute inset-0 h-full w-full object-cover"
      />
    </div>
  );
}

/** Audio gets its own visual identity rather than a video frame with no picture. */
function AudioWaveform() {
  const bars = [
    18, 34, 26, 48, 62, 40, 72, 54, 88, 66, 44, 78, 58, 92, 70, 46, 84, 60, 38, 52, 30,
    64, 42, 24,
  ];
  return (
    <div
      aria-hidden="true"
      className="absolute inset-x-8 top-1/2 flex -translate-y-1/2 items-end justify-center gap-[3px] opacity-45"
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
