"use client";

import { Icon } from "./Icon";
import { cn } from "@/lib/cn";

/**
 * Media input affordance.
 *
 * ⚠️ MOCK — no file is read, uploaded or stored. Direct-to-storage signed
 * uploads arrive at M3.05. This exists so the client can approve how media
 * input reads for the workflows that need it (Image to Video, Video to Video,
 * Extend Video, Music Video).
 */
export function Dropzone({
  kind,
  className,
  onClick,
}: {
  /** Reads inside "Drop {kind} here" — e.g. "an image". */
  kind: string;
  className?: string;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-zx-md hover:border-zx-border-active hover:bg-zx-primary/4 w-full cursor-pointer border-[1.5px] border-dashed border-white/16 px-4 py-5 text-center transition-colors duration-150",
        className,
      )}
    >
      <div className="text-zx-accent mb-[7px] flex justify-center">
        <Icon name="upload" size={20} />
      </div>
      <div className="text-zx-text text-[12.5px] font-bold">
        Drop {kind} here
      </div>
      <div className="text-zx-text-muted mt-[3px] text-[11.5px]">
        or browse your media library
      </div>
    </button>
  );
}
