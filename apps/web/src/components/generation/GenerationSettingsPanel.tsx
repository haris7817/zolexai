"use client";

import { useCreatorParams } from "@/features/workflows/useWorkflowParams";
import type { WorkflowDefinition } from "@/features/workflows/types";
import { WorkflowSelector } from "@/components/workflow/WorkflowSelector";
import { Button } from "@/components/ui/Button";
import { Dropzone } from "@/components/ui/Dropzone";
import { Icon } from "@/components/ui/Icon";
import {
  SectionLabel,
  OptionChip,
  SegmentedControl,
  RangeField,
  ToggleField,
} from "@/components/ui/Controls";
import { featureFlags } from "@/config/feature-flags";
import { cn } from "@/lib/cn";

/** The scripted walkthrough prompt (guide §7 Step 3). */
const EXAMPLE_PROMPT =
  "A cinematic drone shot flying through a futuristic city at sunset, soft purple neon lights, realistic reflections on wet streets, smooth camera movement and cinematic atmosphere.";

/**
 * The right-hand settings panel — the workflow-driven core of the workspace.
 *
 * Every section below is rendered from workflow metadata, never from a branch
 * on workflow id. Switching to Music removes the aspect-ratio and quality
 * sections and swaps the durations, because that workflow's registry entry says
 * so. This is what makes ZolexAI extensible without redesigning the app, and is
 * the single most valuable thing to demonstrate (guide §7 Step 2).
 */
export function GenerationSettingsPanel({
  workflow,
  onGenerate,
  onRequestClose,
  isCompact,
  generateLabel,
}: {
  workflow: WorkflowDefinition;
  onGenerate: () => void;
  onRequestClose: () => void;
  isCompact: boolean;
  generateLabel: string;
}) {
  const params = useCreatorParams();
  const canGenerate = params.prompt.trim().length > 0;

  const hasAspects = workflow.supportedAspectRatios.length > 0;
  const hasQuality =
    workflow.settings.quality && workflow.supportedQualityLevels.length > 0;
  const hasAdvanced =
    workflow.settings.motionStrength ||
    workflow.settings.promptAdherence ||
    workflow.settings.seed;

  return (
    <>
      <div className="flex-1 px-5 pt-5">
        {/* Compact-mode chrome: grab handle (mobile) + close button */}
        <div className="laptop:hidden">
          <div
            aria-hidden="true"
            className="mx-auto mb-[14px] h-1 w-10 rounded-sm bg-white/16 tablet:hidden"
          />
          <div className="mb-4 flex items-center justify-between">
            <span className="text-zx-text text-[15px] font-extrabold">
              Settings
            </span>
            <button
              type="button"
              onClick={onRequestClose}
              aria-label="Close settings"
              className="bg-zx-surface border-zx-border text-zx-text-secondary rounded-zx-sm flex h-9 w-9 cursor-pointer items-center justify-center border"
            >
              <Icon name="close" size={16} />
            </button>
          </div>
        </div>

        <SectionLabel>Workflow</SectionLabel>
        <WorkflowSelector
          activeId={workflow.id}
          onSelect={isCompact ? onRequestClose : undefined}
        />

        {workflow.mediaRequirements ? (
          <>
            <SectionLabel>{workflow.mediaRequirements.label}</SectionLabel>
            <Dropzone
              kind={workflow.mediaRequirements.kind}
              className="mb-6"
            />
          </>
        ) : null}

        {/* ── Prompt ─────────────────────────────────────────────────── */}
        <div className="mb-[10px] flex items-baseline justify-between gap-3">
          <SectionLabel as="label" htmlFor="zx-prompt" className="mb-0">
            Prompt
          </SectionLabel>

          {/* Demo helper — NOT product UI. Lets the presenter drop in the
              scripted prompt instantly while the workspace still opens empty.
              Removed at M1 by flipping featureFlags.demoHelpers. */}
          {featureFlags.demoHelpers ? (
            <button
              type="button"
              onClick={() => params.setPrompt(EXAMPLE_PROMPT)}
              className="text-zx-text-muted hover:text-zx-text-secondary cursor-pointer text-[11.5px] font-bold whitespace-nowrap transition-colors duration-150"
            >
              Use example prompt
            </button>
          ) : null}
        </div>

        <textarea
          id="zx-prompt"
          value={params.prompt}
          onChange={(event) => params.setPrompt(event.target.value)}
          placeholder={workflow.promptPlaceholder}
          rows={4}
          className="bg-zx-surface border-zx-border text-zx-text rounded-zx-md focus:border-zx-border-active mb-6 w-full resize-none border px-[13px] py-3 text-[13px] leading-[1.55] outline-none transition-colors duration-150"
        />

        {/* ── Aspect ratio — hidden entirely for audio output ─────────── */}
        {hasAspects ? (
          <>
            <SectionLabel>Aspect ratio</SectionLabel>
            <div
              role="group"
              aria-label="Aspect ratio"
              className="mb-6 flex gap-[7px]"
            >
              {workflow.supportedAspectRatios.map((ratio) => (
                <OptionChip
                  key={ratio}
                  selected={ratio === params.aspect}
                  onClick={() => params.setAspect(ratio)}
                  className="flex-1 px-1 py-[9px] text-[11.5px]"
                >
                  {ratio}
                </OptionChip>
              ))}
            </div>
          </>
        ) : null}

        {/* ── Duration — the list changes per workflow ────────────────── */}
        <SectionLabel>Duration</SectionLabel>
        <div role="group" aria-label="Duration" className="mb-6 flex flex-wrap gap-[7px]">
          {workflow.supportedDurations.map((duration) => (
            <OptionChip
              key={duration}
              selected={duration === params.duration}
              onClick={() => params.setDuration(duration)}
              className="px-4 py-[9px] text-[12px]"
            >
              {duration}
            </OptionChip>
          ))}
        </div>

        {/* ── Quality — absent when the workflow exposes none ─────────── */}
        {hasQuality ? (
          <>
            <SectionLabel>Quality</SectionLabel>
            <div className="mb-6">
              <SegmentedControl
                label="Quality"
                value={params.quality}
                onChange={params.setQuality}
                options={workflow.supportedQualityLevels.map((level) => ({
                  value: level,
                  label: level,
                }))}
              />
            </div>
          </>
        ) : null}

        {/* ── Advanced ───────────────────────────────────────────────── */}
        {hasAdvanced ? (
          <>
            <button
              type="button"
              onClick={params.toggleAdvanced}
              aria-expanded={params.advancedOpen}
              className="text-zx-text-secondary hover:text-zx-text flex w-full cursor-pointer items-center justify-between pb-[14px] text-[12.5px] font-bold transition-colors duration-150"
            >
              <span className="inline-flex items-center gap-[7px]">
                <Icon name="sliders" size={15} />
                Advanced settings
              </span>
              <span
                className={cn(
                  "text-zx-text-muted flex transition-transform duration-200",
                  params.advancedOpen && "rotate-180",
                )}
              >
                <Icon name="chevron" size={15} />
              </span>
            </button>

            {params.advancedOpen ? (
              <div className="animate-zx-fade-up mb-[18px] flex flex-col gap-[14px]">
                {workflow.settings.motionStrength ? (
                  <RangeField
                    id="zx-motion"
                    label="Motion strength"
                    value={params.motionStrength}
                    onChange={params.setMotionStrength}
                  />
                ) : null}

                {workflow.settings.promptAdherence ? (
                  <RangeField
                    id="zx-adherence"
                    label="Prompt adherence"
                    value={params.promptAdherence}
                    onChange={params.setPromptAdherence}
                  />
                ) : null}

                {workflow.settings.seed ? (
                  <ToggleField
                    id="zx-seed"
                    label="Seed lock"
                    checked={params.seedLocked}
                    onChange={params.toggleSeedLocked}
                  />
                ) : null}
              </div>
            ) : null}
          </>
        ) : null}
      </div>

      {/* ── Generate ─────────────────────────────────────────────────── */}
      <div className="sticky bottom-0 bg-[linear-gradient(transparent,var(--zx-bg-secondary)_30%)] px-5 pt-4 pb-5">
        <Button
          variant="primary"
          size="lg"
          fullWidth
          onClick={onGenerate}
          disabled={!canGenerate}
          aria-disabled={!canGenerate}
        >
          {generateLabel}
        </Button>
      </div>
    </>
  );
}
