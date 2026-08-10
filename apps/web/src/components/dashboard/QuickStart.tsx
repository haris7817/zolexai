"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useCreatorParams } from "@/features/workflows/useWorkflowParams";
import type { WorkflowId } from "@/features/workflows/types";
import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

/**
 * The dashboard's quick-create box.
 *
 * Hands the typed prompt to the creator store and routes to the matching
 * workspace, so the dashboard genuinely starts a creation rather than being a
 * decorative panel — the demo should feel connected (PRE-M1 directive:
 * "navigation should not lead to meaningless dead ends").
 */

const MODES: { label: string; workflowId: WorkflowId }[] = [
  { label: "Video", workflowId: "text-to-video" },
  { label: "Music", workflowId: "music" },
  { label: "Music Video", workflowId: "music-video" },
];

export function QuickStart() {
  const [mode, setMode] = useState(MODES[0]);
  const [prompt, setPrompt] = useState("");
  const setCreatorPrompt = useCreatorParams((state) => state.setPrompt);
  const router = useRouter();

  const start = () => {
    setCreatorPrompt(prompt.trim());
    router.push(`/app/create/${mode.workflowId}`);
  };

  return (
    <div className="relative mb-11">
      {/* Gradient hairline border from the approved design */}
      <div
        aria-hidden="true"
        className="absolute -inset-px rounded-[19px] bg-[linear-gradient(120deg,rgba(157,107,255,0.55),rgba(109,61,245,0.12)_45%,rgba(157,107,255,0.35))]"
      />
      <div className="bg-zx-bg-alt relative rounded-[18px] px-5 pt-5 pb-4">
        <label htmlFor="zx-quick-prompt" className="sr-only">
          Describe what you want to create
        </label>
        <textarea
          id="zx-quick-prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Describe a video, a song, a scene… anything."
          rows={2}
          className="text-zx-text w-full resize-none border-none bg-transparent text-[16px] leading-[1.5] outline-none"
        />

        <div className="mt-[10px] flex flex-wrap items-center justify-between gap-3">
          <div
            role="group"
            aria-label="Creation type"
            className="flex flex-wrap gap-2"
          >
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
            className="rounded-zx-md shadow-zx-cta inline-flex cursor-pointer items-center gap-2 bg-[image:var(--zx-gradient-primary)] px-[26px] py-[11px] text-[14px] font-extrabold text-white transition-[filter] duration-150 hover:brightness-110"
          >
            Generate
            <Icon name="arrowRight" size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}
