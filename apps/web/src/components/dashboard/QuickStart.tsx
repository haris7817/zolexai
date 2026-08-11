"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

/**
 * The dashboard's quick-create box.
 *
 * Carries the typed prompt to the workspace in the URL rather than through a
 * shared store. That is a deliberate improvement: the resulting link is
 * shareable and bookmarkable, it survives a refresh, and it removes the last
 * reason for cross-page client state to exist at all.
 */

const MODES: { label: string; workflowId: string }[] = [
  { label: "Video", workflowId: "text-to-video" },
  { label: "Music", workflowId: "music" },
  { label: "Music Video", workflowId: "music-video" },
];

export function QuickStart() {
  const [mode, setMode] = useState(MODES[0]);
  const [prompt, setPrompt] = useState("");
  const router = useRouter();

  const start = () => {
    const trimmed = prompt.trim();
    const query = trimmed ? `?prompt=${encodeURIComponent(trimmed)}` : "";
    router.push(`/app/create/${mode.workflowId}${query}`);
  };

  return (
    <div className="relative mb-11">
      {/* Gradient hairline border from the approved design */}
      <div
        aria-hidden="true"
        className="absolute -inset-px rounded-[19px] bg-[linear-gradient(120deg,rgba(198,242,36,0.55),rgba(190,242,8,0.12)_45%,rgba(198,242,36,0.35))]"
      />
      <div className="bg-zx-bg-alt relative rounded-[18px] px-5 pt-5 pb-4">
        <label htmlFor="zx-quick-prompt" className="sr-only">
          Describe what you want to create
        </label>
        <textarea
          id="zx-quick-prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            // Enter submits, Shift+Enter adds a line — the convention for a
            // one-line-intent box that happens to be a textarea.
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              start();
            }
          }}
          placeholder="Describe a video, a song, a scene… anything."
          rows={2}
          className="text-zx-text w-full resize-none border-none bg-transparent text-[16px] leading-[1.5] outline-none"
        />

        <div className="mt-[10px] flex flex-wrap items-center justify-between gap-3">
          <div role="group" aria-label="Creation type" className="flex flex-wrap gap-2">
            {MODES.map((option) => {
              const selected = option.label === mode.label;
              return (
                <button
                  key={option.label}
                  type="button"
                  aria-pressed={selected}
                  onClick={() => setMode(option)}
                  className={cn(
                    "cursor-pointer rounded-full border px-[15px] py-[7px] text-[12.5px] font-bold whitespace-nowrap transition-colors duration-150",
                    selected
                      ? "border-zx-border-active bg-zx-primary/25 text-zx-primary-light"
                      : "border-zx-border text-zx-text-muted hover:text-zx-text-secondary bg-transparent",
                  )}
                >
                  {option.label}
                </button>
              );
            })}
          </div>

          <button
            type="button"
            onClick={start}
            className="rounded-zx-md shadow-zx-cta text-zx-on-primary inline-flex cursor-pointer items-center gap-2 bg-[image:var(--zx-gradient-primary)] px-[26px] py-[11px] text-[14px] font-extrabold transition-[filter] duration-150 hover:brightness-110"
          >
            Generate
            <Icon name="arrowRight" size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}
