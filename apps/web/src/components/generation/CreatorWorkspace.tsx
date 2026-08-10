"use client";

import { useCallback, useEffect, useState } from "react";
import { WORKFLOWS } from "@/features/workflows/registry";
import type { WorkflowDefinition } from "@/features/workflows/types";
import { useCreatorParams } from "@/features/workflows/useWorkflowParams";
import {
  useGenerationJobs,
  selectSelectedJob,
} from "@/features/generation/useGenerationJobs";
import { isRunning } from "@/features/generation/types";
import { useIsCompact } from "@/hooks/useBreakpoint";
import { useBodyScrollLock } from "@/hooks/useBodyScrollLock";
import { useEscapeKey } from "@/hooks/useEscapeKey";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import {
  EmptyGenerationState,
  GenerationProgress,
  GenerationResult,
} from "./CreatorCanvas";
import { GenerationSettingsPanel } from "./GenerationSettingsPanel";
import { JobStrip } from "./JobStrip";
import { cn } from "@/lib/cn";

/**
 * ===========================================================================
 * Creator workspace — the approved primary screen
 * ===========================================================================
 *
 * Layout by mode:
 *   desktop 1440+  sidebar 224 · canvas · panel 320   (inline)
 *   laptop  1024+  sidebar 200 · canvas · panel 292   (inline)
 *   tablet  768+   icon rail 64 · canvas · panel as a 340px right drawer
 *   mobile  <768   drawer nav · canvas · panel as a 78vh bottom sheet
 *
 * All of that is CSS. `useIsCompact()` is consulted only for the panel's
 * `role`/`aria-modal` and whether Escape and scroll-lock apply — things CSS
 * cannot express. See hooks/useBreakpoint.ts.
 */
export function CreatorWorkspace({
  workflow,
}: {
  workflow: WorkflowDefinition;
}) {
  const [panelOpen, setPanelOpen] = useState(false);
  const isCompact = useIsCompact();

  const params = useCreatorParams();
  const syncWorkflow = useCreatorParams((state) => state.syncWorkflow);

  const createJob = useGenerationJobs((state) => state.createJob);
  const selectedJob = useGenerationJobs(selectSelectedJob);

  /**
   * Settings preservation on workflow switch. Keeps duration / aspect / quality
   * when the incoming workflow supports them, else falls back to its first
   * supported value. Verify with: Text to Video @ 15s → Image to Video → 5s.
   */
  useEffect(() => {
    syncWorkflow(workflow);
  }, [workflow, syncWorkflow]);

  const closePanel = useCallback(() => setPanelOpen(false), []);
  const overlayOpen = isCompact && panelOpen;
  useBodyScrollLock(overlayOpen);
  useEscapeKey(overlayOpen, closePanel);

  const handleGenerate = useCallback(() => {
    const state = useCreatorParams.getState();
    if (!state.prompt.trim()) return;

    createJob({
      workflow,
      prompt: state.prompt.trim(),
      parameters: {
        duration: state.duration,
        aspectRatio: state.aspect,
        quality: state.quality,
        motionStrength: state.motionStrength,
        promptAdherence: state.promptAdherence,
        seed: state.seedLocked ? 123456 : null,
      },
    });
    setPanelOpen(false);
  }, [createJob, workflow]);

  /** Prefills the creator from a previous job — never mutates that job. */
  const handleReuseSettings = useCallback(() => {
    if (!selectedJob) return;
    const { parameters, prompt } = selectedJob;
    params.setPrompt(prompt);
    params.setDuration(parameters.duration);
    if (parameters.aspectRatio) params.setAspect(parameters.aspectRatio);
    if (parameters.quality) params.setQuality(parameters.quality);
    params.setMotionStrength(parameters.motionStrength);
    params.setPromptAdherence(parameters.promptAdherence);
    if (isCompact) setPanelOpen(true);
  }, [selectedJob, params, isCompact]);

  const jobWorkflow: WorkflowDefinition = selectedJob
    ? WORKFLOWS[selectedJob.workflowId]
    : workflow;

  const jobRunning = selectedJob ? isRunning(selectedJob.status) : false;
  const showResult = selectedJob?.status === "Completed";
  const generateLabel = jobRunning ? "Generate Another" : "Generate";
  const canGenerate = params.prompt.trim().length > 0;

  return (
    <div className="flex min-h-0 flex-1">
      {/* ── Canvas column ──────────────────────────────────────────── */}
      <main className="flex min-w-0 flex-1 flex-col px-4 pt-4 pb-6 tablet:min-h-0 tablet:overflow-y-auto tablet:px-[26px] tablet:py-[22px]">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-zx-text m-0 text-[19px] font-extrabold tracking-[-0.02em]">
              {workflow.name}
            </h1>
            <p className="text-zx-text-secondary mt-[2px] text-[12.5px]">
              {workflow.description}
            </p>
          </div>

          <div className="hidden gap-2 tablet:flex">
            <Button variant="ghost" size="sm" onClick={noop}>
              <Icon name="history" size={15} />
              History
            </Button>
            <Button variant="ghost" size="sm" onClick={noop}>
              <Icon name="share" size={15} />
              Share
            </Button>
          </div>
        </div>

        <div className="rounded-zx-lg border-zx-border relative flex min-h-[52vh] flex-1 items-center justify-center overflow-hidden border bg-[radial-gradient(ellipse_at_50%_18%,rgba(109,61,245,0.07),transparent_60%),var(--zx-bg-secondary)] tablet:min-h-0">
          {!selectedJob ? <EmptyGenerationState workflow={workflow} /> : null}
          {selectedJob && jobRunning ? (
            <GenerationProgress job={selectedJob} />
          ) : null}
          {selectedJob && showResult ? (
            <GenerationResult
              job={selectedJob}
              workflow={jobWorkflow}
              onReuseSettings={handleReuseSettings}
              onVariation={handleGenerate}
            />
          ) : null}
        </div>

        <JobStrip />

        {/* Compact-mode action bar — the panel is an overlay here, so Generate
            must stay reachable without opening it. */}
        <div className="sticky bottom-3 z-20 mt-4 flex gap-[10px] laptop:hidden">
          <Button
            variant="ghost"
            size="lg"
            onClick={() => setPanelOpen(true)}
            aria-expanded={panelOpen}
            className="bg-zx-surface-elevated text-zx-text flex-1"
          >
            <Icon name="settings" size={16} />
            Settings
          </Button>
          <Button
            variant="primary"
            size="lg"
            onClick={handleGenerate}
            disabled={!canGenerate}
            aria-disabled={!canGenerate}
            className="flex-2"
          >
            {generateLabel}
          </Button>
        </div>
      </main>

      {/* ── Settings panel ─────────────────────────────────────────── */}
      <aside
        // Modal only while it is an overlay; a plain complementary region when inline.
        role={isCompact ? "dialog" : undefined}
        aria-modal={isCompact ? true : undefined}
        aria-label="Generation settings"
        className={cn(
          "bg-zx-bg-alt box-border flex-col overflow-y-auto",
          // mobile: bottom sheet
          "border-zx-border fixed inset-x-0 bottom-0 z-40 max-h-[78vh] rounded-t-[18px] border-t shadow-[0_-20px_60px_rgba(0,0,0,0.5)]",
          // tablet: right drawer
          "tablet:inset-y-0 tablet:right-0 tablet:left-auto tablet:w-[340px] tablet:max-h-none tablet:rounded-t-none tablet:border-t-0 tablet:border-l tablet:shadow-[-24px_0_60px_rgba(0,0,0,0.45)]",
          // laptop+: inline column, no longer an overlay
          "laptop:static laptop:z-auto laptop:w-[292px] laptop:shrink-0 laptop:shadow-none desktop:w-[320px]",
          panelOpen ? "flex" : "hidden laptop:flex",
        )}
      >
        <GenerationSettingsPanel
          workflow={workflow}
          onGenerate={handleGenerate}
          onRequestClose={closePanel}
          isCompact={isCompact}
          generateLabel={generateLabel}
        />
      </aside>

      {overlayOpen ? (
        <div
          onClick={closePanel}
          aria-hidden="true"
          className="fixed inset-0 z-39 bg-black/55 laptop:hidden"
        />
      ) : null}
    </div>
  );
}

/** Present for design review; wired up in later milestones. */
function noop() {
  /* intentionally empty */
}
