"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { isRunning, type Workflow } from "@zolexai/workflow-contracts";
import { ApiError } from "@/lib/api/client";
import { useIsCompact } from "@/hooks/useBreakpoint";
import { useBodyScrollLock } from "@/hooks/useBodyScrollLock";
import { useEscapeKey } from "@/hooks/useEscapeKey";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import { useWorkflow } from "@/features/workflows/queries";
import {
  useCancelGeneration,
  useCreateGeneration,
  useGeneration,
  useRecentGenerations,
} from "@/features/generation/queries";
import { useGenerationStream } from "@/features/generation/useGenerationStream";
import {
  buildGenerationSchema,
  preserveValues,
  toCreateInput,
  valuesFromJob,
  withSourceAsset,
  type GenerationFormValues,
} from "@/features/generation/form";
import {
  EmptyGenerationState,
  GenerationFailed,
  GenerationProgress,
  GenerationResult,
} from "./CreatorCanvas";
import { GenerationSettingsPanel } from "./GenerationSettingsPanel";
import { JobStrip } from "./JobStrip";
import { cn } from "@/lib/cn";

/**
 * ===========================================================================
 * Creator workspace — the approved primary screen, now on the real backend
 * ===========================================================================
 *
 * Layout by mode (unchanged from the approved design):
 *   desktop 1440+  sidebar 224 · canvas · panel 320   (inline)
 *   laptop  1024+  sidebar 200 · canvas · panel 292   (inline)
 *   tablet  768+   icon rail 64 · canvas · panel as a 340px right drawer
 *   mobile  <768   drawer nav · canvas · panel as a 78vh bottom sheet
 *
 * All of that is CSS. `useIsCompact()` is consulted only for the panel's
 * `role`/`aria-modal` and whether Escape and scroll-lock apply — things CSS
 * cannot express.
 *
 * ## What changed at M1
 *
 * Form state is React Hook Form with a Zod schema **generated from the workflow
 * definition**, so validation follows the tool rather than being written per
 * tool. Job state is React Query, written to by SSE. The zustand store, the
 * timer-based pipeline and the mock job list are gone.
 *
 * The one piece of genuinely local state left is `selectedJobId` — which result
 * this tab is looking at. That is a view preference, not data, so it belongs
 * here and nowhere else.
 */
export function CreatorWorkspace({
  workflowId,
  initialWorkflow,
  initialPrompt = "",
  initialSourceAssetId = null,
}: {
  workflowId: string;
  /** Server-rendered from the YAML catalogue, so the panel paints complete. */
  initialWorkflow: Workflow;
  /** From `?prompt=` — the dashboard's quick-create box hands it over this way. */
  initialPrompt?: string;
  /** From `?source=` — Extend hands over the result being continued. */
  initialSourceAssetId?: string | null;
}) {
  const workflow = useWorkflow(workflowId) ?? initialWorkflow;

  const [panelOpen, setPanelOpen] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const isCompact = useIsCompact();

  const schema = useMemo(() => buildGenerationSchema(workflow), [workflow]);
  const form = useForm<GenerationFormValues>({
    resolver: zodResolver(schema),
    defaultValues: withSourceAsset(
      workflow,
      { ...preserveValues(workflow, undefined), prompt: initialPrompt },
      initialSourceAssetId,
    ),
    mode: "onChange",
  });

  /**
   * Settings preservation on workflow switch.
   *
   * Keeps duration / aspect / quality when the incoming workflow supports them,
   * else falls back to its first supported value. Verified by test with:
   * Text to Video @ 15s → Image to Video → 5s.
   *
   * `getValues()` rather than a watched value, so this runs on workflow change
   * only and never re-enters on every keystroke.
   */
  const lastWorkflowId = useRef(workflow.id);
  useEffect(() => {
    if (lastWorkflowId.current === workflow.id) return;
    lastWorkflowId.current = workflow.id;
    form.reset(preserveValues(workflow, form.getValues()));
    setSubmitError(null);
  }, [workflow, form]);

  const closePanel = useCallback(() => setPanelOpen(false), []);
  const overlayOpen = isCompact && panelOpen;
  useBodyScrollLock(overlayOpen);
  useEscapeKey(overlayOpen, closePanel);

  // ── Jobs ──────────────────────────────────────────────────────────────

  const { jobs: recentJobs } = useRecentGenerations();

  // Default to the newest job so a refresh mid-generation lands back on it
  // rather than on an empty canvas.
  useEffect(() => {
    if (selectedJobId === null && recentJobs.length > 0) {
      setSelectedJobId(recentJobs[0].id);
    }
  }, [recentJobs, selectedJobId]);

  const { data: selectedJob } = useGeneration(selectedJobId);
  const jobRunning = selectedJob ? isRunning(selectedJob.status) : false;

  // One stream, only while the watched job is unfinished.
  useGenerationStream(selectedJobId, { enabled: jobRunning });

  const createGeneration = useCreateGeneration();
  const cancelGeneration = useCancelGeneration();

  const submit = form.handleSubmit(async (values) => {
    setSubmitError(null);
    try {
      const accepted = await createGeneration.mutateAsync(toCreateInput(workflow, values));
      setSelectedJobId(accepted.job_id);
      setPanelOpen(false);
    } catch (cause) {
      if (cause instanceof ApiError) {
        // Field-level problems the API reported are attached to the matching
        // control, so the user sees them where they can act on them.
        for (const { field, reason } of cause.fieldErrors) {
          const name = FIELD_MAP[field] ?? field;
          form.setError(name as keyof GenerationFormValues, { message: reason });
        }
        setSubmitError(cause.fieldErrors.length ? null : cause.message);
      } else {
        setSubmitError("Your generation could not be submitted. Please try again.");
      }
    }
  });

  /** Prefills the creator from a job — never mutates that job. */
  const reuseSettings = useCallback(() => {
    if (!selectedJob) return;
    form.reset(
      valuesFromJob(workflow, selectedJob.prompt, selectedJob.parameters as Record<string, unknown>),
    );
    if (isCompact) setPanelOpen(true);
  }, [selectedJob, workflow, form, isCompact]);

  const canGenerate = form.formState.isValid && !createGeneration.isPending;
  const generateLabel = createGeneration.isPending
    ? "Submitting…"
    : jobRunning
      ? "Generate Another"
      : "Generate";

  return (
    <FormProvider {...form}>
      <form onSubmit={submit} className="flex min-h-0 flex-1">
        {/* ── Canvas column — a section, not <main>: the app shell owns the
            single main landmark and nesting a second one is invalid. */}
        <section className="tablet:min-h-0 tablet:overflow-y-auto tablet:px-[26px] tablet:py-[22px] flex min-w-0 flex-1 flex-col px-4 pt-4 pb-6">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-zx-text m-0 text-[19px] font-extrabold tracking-[-0.02em]">
                {workflow.name}
              </h1>
              <p className="text-zx-text-secondary mt-[2px] text-[12.5px]">
                {workflow.description}
              </p>
            </div>
          </div>

          <div className="rounded-zx-lg border-zx-border relative flex min-h-[52vh] flex-1 items-center justify-center overflow-hidden border bg-[radial-gradient(ellipse_at_50%_18%,rgba(190,242,8,0.07),transparent_60%),var(--zx-bg-secondary)] tablet:min-h-0">
            {!selectedJob ? <EmptyGenerationState workflow={workflow} /> : null}

            {selectedJob && jobRunning ? (
              <GenerationProgress
                job={selectedJob}
                onCancel={() => cancelGeneration.mutate(selectedJob.id)}
                cancelling={cancelGeneration.isPending}
              />
            ) : null}

            {selectedJob?.status === "completed" ? (
              <GenerationResult
                job={selectedJob}
                workflow={workflow}
                onReuseSettings={reuseSettings}
                onVariation={() => void submit()}
              />
            ) : null}

            {selectedJob &&
            (selectedJob.status === "failed" || selectedJob.status === "cancelled") ? (
              <GenerationFailed job={selectedJob} onRetry={reuseSettings} />
            ) : null}
          </div>

          {submitError ? (
            <p
              role="alert"
              className="border-zx-error/40 bg-zx-error/8 text-zx-error rounded-zx-md mt-3 border px-3 py-2 text-[12.5px] font-semibold"
            >
              {submitError}
            </p>
          ) : null}

          <JobStrip selectedJobId={selectedJobId} onSelect={setSelectedJobId} />

          {/* Compact-mode action bar — the panel is an overlay here, so
              Generate must stay reachable without opening it. */}
          <div className="laptop:hidden sticky bottom-3 z-20 mt-4 flex gap-[10px]">
            <Button
              type="button"
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
              type="submit"
              variant="primary"
              size="lg"
              disabled={!canGenerate}
              aria-disabled={!canGenerate}
              className="flex-2"
            >
              {generateLabel}
            </Button>
          </div>
        </section>

        {/* ── Settings panel ───────────────────────────────────────── */}
        <aside
          // Modal only while it is an overlay; a plain complementary region
          // when inline.
          role={isCompact ? "dialog" : undefined}
          aria-modal={isCompact ? true : undefined}
          aria-label="Generation settings"
          className={cn(
            "bg-zx-bg-alt box-border flex-col overflow-y-auto",
            "border-zx-border fixed inset-x-0 bottom-0 z-40 max-h-[78vh] rounded-t-[18px] border-t shadow-[0_-20px_60px_rgba(0,0,0,0.5)]",
            "tablet:inset-y-0 tablet:right-0 tablet:left-auto tablet:max-h-none tablet:w-[340px] tablet:rounded-t-none tablet:border-t-0 tablet:border-l tablet:shadow-[-24px_0_60px_rgba(0,0,0,0.45)]",
            "laptop:static laptop:z-auto laptop:w-[292px] laptop:shrink-0 laptop:shadow-none desktop:w-[320px]",
            panelOpen ? "flex" : "laptop:flex hidden",
          )}
        >
          <GenerationSettingsPanel
            workflow={workflow}
            onRequestClose={closePanel}
            isCompact={isCompact}
            generateLabel={generateLabel}
            canGenerate={canGenerate}
          />
        </aside>

        {overlayOpen ? (
          <div
            onClick={closePanel}
            aria-hidden="true"
            className="laptop:hidden fixed inset-0 z-39 bg-black/55"
          />
        ) : null}
      </form>
    </FormProvider>
  );
}

/** API field names → form field names. */
const FIELD_MAP: Record<string, string> = {
  duration: "duration",
  aspect_ratio: "aspectRatio",
  quality: "quality",
  prompt: "prompt",
};
