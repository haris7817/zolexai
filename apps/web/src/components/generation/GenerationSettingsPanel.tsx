"use client";

import { useState } from "react";
import { Controller, useFormContext } from "react-hook-form";
import type { Workflow } from "@zolexai/workflow-contracts";
import { WorkflowSelector } from "@/components/workflow/WorkflowSelector";
import { Button } from "@/components/ui/Button";
import { Dropzone } from "@/components/ui/Dropzone";
import { Icon } from "@/components/ui/Icon";
import {
  SectionLabel,
  OptionChip,
  SegmentedControl,
  RangeField,
  SelectField,
  ToggleField,
} from "@/components/ui/Controls";
import {
  DIALOGUE_LANGUAGES,
  LYRIC_LANGUAGES,
  type GenerationFormValues,
} from "@/features/generation/form";
import {
  autoDurationLabel,
  durationLabel,
  durationsForQuality,
  hasAdvancedSettings,
  qualityLabel,
  showsAspectRatio,
  showsQuality,
  showsSound,
} from "@/services/workflows";
import { cn } from "@/lib/cn";

/**
 * The right-hand settings panel — the workflow-driven core of the workspace.
 *
 * Every section is rendered from workflow metadata served by the API, never
 * from a branch on workflow id. Switching to Music removes the aspect-ratio and
 * quality sections and swaps the durations because that workflow's definition
 * says so. Video to Video shows a required source video AND an optional
 * reference image for the same reason (directive §14).
 *
 * This is what makes ZolexAI extensible without redesigning the app: a workflow
 * added in M2 renders correctly here with no change to this file.
 */
export function GenerationSettingsPanel({
  workflow,
  onRequestClose,
  isCompact,
  generateLabel,
  canGenerate,
}: {
  workflow: Workflow;
  onRequestClose: () => void;
  isCompact: boolean;
  generateLabel: string;
  canGenerate: boolean;
}) {
  const form = useFormContext<GenerationFormValues>();
  const { errors } = form.formState;

  /**
   * Whether the advanced section is expanded — component state, not a form
   * field. It is a disclosure preference, not part of the submitted request,
   * and putting it in the form would round-trip it through validation and
   * `reset()` on every workflow switch.
   */
  const [showAdvanced, setShowAdvanced] = useState(false);

  /** Only meaningful on workflows that declare `settings.prompt_modes`. */
  const directorMode =
    workflow.settings.prompt_modes && form.watch("promptMode") === "director";

  /**
   * The selected quality level narrows what the other controls offer
   * (Fast/Best, 27 Aug: Best's engine sells 5-30s and generates native
   * audio; Fast is the full ladder with no sound choice).
   */
  const selectedQuality = form.watch("quality");
  const offeredDurations = durationsForQuality(workflow, selectedQuality);

  /**
   * Director mode on an image-fed workflow is source-anchored: the upload
   * decides who and what is in the scene, the idea decides what happens.
   * The copy says so — promising "AI will create the scene" over a photo
   * the user chose would be promising the wrong thing.
   */
  const directorAnchored = workflow.inputs.some(
    (input) => input.kind === "image" && input.required,
  );

  return (
    <>
      <div className="flex-1 px-5 pt-5">
        {/* Compact-mode chrome: grab handle (mobile) + close button */}
        <div className="laptop:hidden">
          <div
            aria-hidden="true"
            className="tablet:hidden mx-auto mb-[14px] h-1 w-10 rounded-sm bg-white/16"
          />
          <div className="mb-4 flex items-center justify-between">
            <span className="text-zx-text text-[15px] font-extrabold">Settings</span>
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

        {/* ── Media inputs — one per declared role ─────────────────── */}
        {workflow.inputs.map((input) => (
          <div key={input.role}>
            <SectionLabel>
              {input.label}
              {!input.required ? (
                <span className="text-zx-text-muted ml-[6px] font-semibold normal-case">
                  optional
                </span>
              ) : null}
            </SectionLabel>
            <Controller
              control={form.control}
              name={`inputs.${input.role}` as const}
              render={({ field }) => (
                <Dropzone
                  input={input}
                  value={(field.value as string | null) ?? null}
                  onChange={field.onChange}
                  className="mb-6"
                />
              )}
            />
          </div>
        ))}

        {/* ── Prompt mode — rendered only when the workflow declares it.
            Standard submits exactly what the pre-feature form submitted;
            Idea (Director) has the backend plan characters, dialogue and
            timing from a simple idea before generating. */}
        {workflow.settings.prompt_modes ? (
          <>
            <SectionLabel>Prompt mode</SectionLabel>
            <Controller
              control={form.control}
              name="promptMode"
              render={({ field }) => (
                <div className={directorMode ? "mb-2" : "mb-6"}>
                  <SegmentedControl
                    label="Prompt mode"
                    value={field.value ?? "standard"}
                    onChange={field.onChange}
                    options={[
                      { value: "standard", label: "Standard" },
                      { value: "director", label: "Idea (Director)" },
                    ]}
                  />
                </div>
              )}
            />
            {directorMode ? (
              <p className="text-zx-text-muted mb-4 text-[11.5px] leading-[1.5]">
                {directorAnchored
                  ? "Describe what should happen — your image defines who and what is in the scene, and AI plans the action and dialogue around it."
                  : "Describe a simple idea — AI will create the scene, characters and dialogue automatically."}
              </p>
            ) : null}
          </>
        ) : null}

        {/* ── Prompt ───────────────────────────────────────────────── */}
        <SectionLabel as="label" htmlFor="zx-prompt">
          {directorMode ? "Idea" : "Prompt"}
        </SectionLabel>
        <textarea
          id="zx-prompt"
          {...form.register("prompt")}
          placeholder={
            directorMode
              ? directorAnchored
                ? "Describe what happens next in your image — who speaks, what they say and do…"
                : "Describe your idea — who is in the scene, where, and what happens…"
              : workflow.prompt.placeholder
          }
          maxLength={workflow.prompt.max_length}
          rows={4}
          aria-invalid={Boolean(errors.prompt)}
          className={cn(
            "bg-zx-surface border-zx-border text-zx-text rounded-zx-md focus:border-zx-border-active w-full resize-none border px-[13px] py-3 text-[13px] leading-[1.55] outline-none transition-colors duration-150",
            errors.prompt ? "border-zx-error/60 mb-[6px]" : "mb-6",
          )}
        />
        {errors.prompt ? (
          <p role="alert" className="text-zx-error mb-6 text-[11.5px] font-semibold">
            {errors.prompt.message}
          </p>
        ) : null}

        {/* ── Dialogue language — Director mode only ───────────────── */}
        {directorMode ? (
          <>
            <SectionLabel as="label" htmlFor="zx-dialogue-language">
              Dialogue language
            </SectionLabel>
            <Controller
              control={form.control}
              name="dialogueLanguage"
              render={({ field }) => (
                <SelectField
                  id="zx-dialogue-language"
                  value={field.value ?? DIALOGUE_LANGUAGES[0]}
                  onChange={(event) => field.onChange(event.target.value)}
                  className="mb-6"
                >
                  {DIALOGUE_LANGUAGES.map((language) => (
                    <option key={language} value={language}>
                      {language === "auto"
                        ? "Auto (match my idea)"
                        : language.charAt(0).toUpperCase() + language.slice(1)}
                    </option>
                  ))}
                </SelectField>
              )}
            />
          </>
        ) : null}

        {/* ── Lyrics — rendered only when the workflow declares the control.
            The customer's own words are never rewritten; empty means the
            platform writes them, and the language choice applies to THAT. */}
        {workflow.settings.lyrics ? (
          <>
            <SectionLabel as="label" htmlFor="zx-lyrics">
              Lyrics
              <span className="text-zx-text-muted ml-[6px] font-semibold normal-case">
                optional
              </span>
            </SectionLabel>
            <textarea
              id="zx-lyrics"
              {...form.register("lyrics")}
              placeholder={
                "Paste your own lyrics in any language — or leave this empty and we'll write them from your prompt."
              }
              rows={4}
              aria-invalid={Boolean(errors.lyrics)}
              className={cn(
                "bg-zx-surface border-zx-border text-zx-text rounded-zx-md focus:border-zx-border-active w-full resize-none border px-[13px] py-3 text-[13px] leading-[1.55] outline-none transition-colors duration-150",
                errors.lyrics ? "border-zx-error/60 mb-[6px]" : "mb-4",
              )}
            />
            {errors.lyrics ? (
              <p role="alert" className="text-zx-error mb-4 text-[11.5px] font-semibold">
                {errors.lyrics.message}
              </p>
            ) : null}

            <SectionLabel as="label" htmlFor="zx-lyrics-language">
              Lyrics language
            </SectionLabel>
            <Controller
              control={form.control}
              name="lyricsLanguage"
              render={({ field }) => (
                <SelectField
                  id="zx-lyrics-language"
                  value={field.value ?? LYRIC_LANGUAGES[0]}
                  onChange={(event) => field.onChange(event.target.value)}
                  className="mb-[6px]"
                >
                  {LYRIC_LANGUAGES.map((language) => (
                    <option key={language} value={language}>
                      {language}
                    </option>
                  ))}
                </SelectField>
              )}
            />
            <p className="text-zx-text-muted mb-6 text-[11.5px] leading-[1.5]">
              Applies when we write the lyrics for you. Your own pasted lyrics
              can be in any language.
            </p>
          </>
        ) : null}

        {/* ── Aspect ratio — hidden entirely for audio output ──────── */}
        {showsAspectRatio(workflow) ? (
          <>
            <SectionLabel>Aspect ratio</SectionLabel>
            <Controller
              control={form.control}
              name="aspectRatio"
              render={({ field }) => (
                <div role="group" aria-label="Aspect ratio" className="mb-6 flex gap-[7px]">
                  {workflow.supported_aspect_ratios.map((ratio) => (
                    <OptionChip
                      key={ratio}
                      selected={ratio === field.value}
                      onClick={() => field.onChange(ratio)}
                      className="flex-1 px-1 py-[9px] text-[11.5px]"
                    >
                      {ratio}
                    </OptionChip>
                  ))}
                </div>
              )}
            />
          </>
        ) : null}

        {/* ── Duration — mode and options both come from the workflow ──
            `fixed` and `minutes` render choice chips (the strings differ,
            the control does not). `source` renders a statement instead of a
            control: the length comes from the uploaded file, and offering
            chips would promise a choice the backend is required to refuse. */}
        <SectionLabel>Duration</SectionLabel>
        {workflow.duration_mode === "source" ? (
          <p
            data-testid="auto-duration"
            className="border-zx-border bg-zx-surface text-zx-text-secondary rounded-zx-md mb-6 flex items-center gap-2 border px-[13px] py-[11px] text-[12.5px] font-semibold"
          >
            <Icon name="clock" size={14} />
            {autoDurationLabel(workflow)}
          </p>
        ) : (
          <Controller
            control={form.control}
            name="duration"
            render={({ field }) => (
              <div
                role="group"
                aria-label="Duration"
                className="mb-6 flex flex-wrap gap-[7px]"
              >
                {offeredDurations.map((duration) => (
                  <OptionChip
                    key={duration}
                    selected={duration === field.value}
                    onClick={() => field.onChange(duration)}
                    className="px-4 py-[9px] text-[12px]"
                  >
                    {durationLabel(duration)}
                  </OptionChip>
                ))}
              </div>
            )}
          />
        )}

        {/* ── Quality — absent when the workflow exposes none ──────── */}
        {showsQuality(workflow) ? (
          <>
            <SectionLabel>Quality</SectionLabel>
            <Controller
              control={form.control}
              name="quality"
              render={({ field }) => (
                <div className="mb-6">
                  <SegmentedControl
                    label="Quality"
                    value={field.value}
                    onChange={(level: string) => {
                      field.onChange(level);
                      // The new level may not offer the selected duration
                      // (Best has no 60s). Snap to the nearest offered value
                      // rather than leaving the form invalid.
                      const offered = durationsForQuality(workflow, level);
                      const current = form.getValues("duration");
                      if (current && !offered.includes(current)) {
                        form.setValue("duration", offered[offered.length - 1] ?? null, {
                          shouldValidate: true,
                        });
                      }
                    }}
                    options={workflow.supported_quality_levels.map((level) => ({
                      value: level,
                      label: qualityLabel(level),
                    }))}
                  />
                </div>
              )}
            />
          </>
        ) : null}

        {/* ── Sound — on every quality level (client round two, 27 Aug):
            both engines deliver an audio track ─────────────────────── */}
        {showsSound(workflow) ? (
          <Controller
            control={form.control}
            name="soundOn"
            render={({ field }) => (
              <div className="mb-6">
                <ToggleField
                  id="zx-sound"
                  label="Sound"
                  checked={field.value}
                  onChange={field.onChange}
                />
              </div>
            )}
          />
        ) : null}

        {/* ── Advanced ─────────────────────────────────────────────── */}
        {hasAdvancedSettings(workflow) ? (
          <>
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              aria-expanded={showAdvanced}
              className="text-zx-text-secondary hover:text-zx-text flex w-full cursor-pointer items-center justify-between pb-[14px] text-[12.5px] font-bold transition-colors duration-150"
            >
              <span className="inline-flex items-center gap-[7px]">
                <Icon name="sliders" size={15} />
                Advanced settings
              </span>
              <span
                className={cn(
                  "text-zx-text-muted flex transition-transform duration-200",
                  showAdvanced && "rotate-180",
                )}
              >
                <Icon name="chevron" size={15} />
              </span>
            </button>

            {showAdvanced ? (
              <div className="animate-zx-fade-up mb-[18px] flex flex-col gap-[14px]">
                {workflow.settings.motion_strength ? (
                  <Controller
                    control={form.control}
                    name="motionStrength"
                    render={({ field }) => (
                      <RangeField
                        id="zx-motion"
                        label="Motion strength"
                        value={field.value}
                        onChange={field.onChange}
                      />
                    )}
                  />
                ) : null}

                {workflow.settings.prompt_adherence ? (
                  <Controller
                    control={form.control}
                    name="promptAdherence"
                    render={({ field }) => (
                      <RangeField
                        id="zx-adherence"
                        label="Prompt adherence"
                        value={field.value}
                        onChange={field.onChange}
                      />
                    )}
                  />
                ) : null}

                {workflow.settings.seed ? (
                  <Controller
                    control={form.control}
                    name="seedLocked"
                    render={({ field }) => (
                      <ToggleField
                        id="zx-seed"
                        label="Seed lock"
                        checked={field.value}
                        onChange={field.onChange}
                      />
                    )}
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
          type="submit"
          variant="primary"
          size="lg"
          fullWidth
          disabled={!canGenerate}
          aria-disabled={!canGenerate}
        >
          {generateLabel}
        </Button>
      </div>
    </>
  );
}
