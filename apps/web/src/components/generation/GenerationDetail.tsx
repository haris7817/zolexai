"use client";

import Link from "next/link";
import { isRunning } from "@zolexai/workflow-contracts";
import { AppPage } from "@/components/ui/PageHeader";
import { Button, ButtonLink } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import { EmptyState, Skeleton, StatusPill } from "@/components/ui/Feedback";
import { MediaPreview } from "@/components/media/MediaPreview";
import { statusTone, relativeTime } from "./GenerationCard";
import { useCancelGeneration, useGeneration } from "@/features/generation/queries";
import { useGenerationStream } from "@/features/generation/useGenerationStream";
import { useWorkflow } from "@/features/workflows/queries";
import { primaryOutput } from "@/services/generations";
import { fetchDownloadUrl } from "@/services/assets";
import { durationLabel } from "@/services/workflows";
import { useState } from "react";

/**
 * One generation, in full.
 *
 * Live: if the job is still running, this page streams its progress over SSE
 * exactly as the workspace does — so a link shared mid-generation shows it
 * finishing rather than a stale snapshot.
 *
 * Result actions come from the workflow's capabilities, so an audio generation
 * shows no Extend here either — the same rule as the workspace, from the same
 * source.
 */
export function GenerationDetail({ generationId }: { generationId: string }) {
  const { data: job, isPending, isError } = useGeneration(generationId);
  const workflow = useWorkflow(job?.workflow_id);
  const cancel = useCancelGeneration();
  const [downloading, setDownloading] = useState(false);

  const running = job ? isRunning(job.status) : false;
  useGenerationStream(generationId, { enabled: running });

  const output = job ? primaryOutput(job) : undefined;
  const isAudio = output?.kind === "audio";

  const download = async () => {
    if (!output) return;
    setDownloading(true);
    try {
      window.location.assign(await fetchDownloadUrl(output.asset_id));
    } finally {
      setDownloading(false);
    }
  };

  return (
    <AppPage>
      <Link
        href="/app/generations"
        className="text-zx-text-secondary hover:text-zx-text mb-5 inline-flex items-center gap-[6px] text-[13px] font-bold"
      >
        <Icon name="chevronLeft" size={15} />
        Back to generations
      </Link>

      {isPending ? (
        <div className="laptop:grid-cols-[1fr_300px] grid grid-cols-1 gap-6">
          <Skeleton className="rounded-[14px]" style={{ aspectRatio: "16 / 9" }} />
          <Skeleton className="rounded-zx-lg h-[320px]" />
        </div>
      ) : isError || !job ? (
        <EmptyState
          icon="alert"
          title="Generation not found"
          description="It may have been removed, or the link may be incorrect."
          action={
            <ButtonLink href="/app/generations" variant="ghost" size="md">
              Back to generations
            </ButtonLink>
          }
        />
      ) : (
        <div className="laptop:grid-cols-[1fr_300px] grid grid-cols-1 gap-6">
          <div>
            <div className="relative">
              <MediaPreview
                url={output?.url ?? null}
                kind={output?.kind ?? "image"}
                aspectRatio={isAudio ? "16 / 7" : "16 / 9"}
                fallbackGradient={workflow?.ui.thumb ?? "linear-gradient(140deg,#1C232A,#0B0E11)"}
                className="border-zx-border rounded-[14px] border"
              />
              <span className="absolute top-3 left-3">
                <StatusPill tone={statusTone(job.status)}>
                  {job.stage_label.toUpperCase()}
                </StatusPill>
              </span>
            </div>

            {running ? (
              <div className="mt-4">
                <div aria-hidden="true" className="h-[5px] overflow-hidden rounded-[3px] bg-white/7">
                  <div
                    className="h-full rounded-[3px] bg-[image:var(--zx-gradient-primary)] transition-[width] duration-400 ease-out"
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
                <div className="mt-2 flex items-center justify-between gap-3">
                  <span role="status" className="text-zx-text-secondary text-[12.5px]">
                    {job.hint || job.stage_label}
                  </span>
                  <button
                    type="button"
                    onClick={() => cancel.mutate(job.id)}
                    disabled={cancel.isPending}
                    className="text-zx-text-muted hover:text-zx-text-secondary cursor-pointer text-[12px] font-bold transition-colors duration-150 disabled:opacity-50"
                  >
                    {cancel.isPending ? "Cancelling…" : "Cancel"}
                  </button>
                </div>
              </div>
            ) : null}

            {job.error ? (
              <div className="border-zx-error/40 bg-zx-error/8 rounded-zx-md mt-4 flex items-start gap-3 border p-4">
                <span className="text-zx-error mt-[1px]">
                  <Icon name="alert" size={16} />
                </span>
                <div>
                  <div className="text-zx-text mb-1 text-[13px] font-bold">
                    Generation failed
                  </div>
                  <p className="text-zx-text-secondary m-0 text-[12.5px] leading-[1.55]">
                    {job.error.message}
                  </p>
                </div>
              </div>
            ) : null}

            {job.status === "completed" && workflow ? (
              <div className="mt-[18px] flex flex-wrap gap-[9px]">
                {workflow.capabilities.download && output ? (
                  <Button variant="primary" size="md" onClick={download} disabled={downloading}>
                    <Icon name="download" size={15} />
                    {downloading ? "Preparing…" : "Download"}
                  </Button>
                ) : null}
                {workflow.capabilities.extend ? (
                  <ButtonLink href="/app/create/extend-video" variant="ghost" size="md">
                    <Icon name="extend" size={15} />
                    Extend
                  </ButtonLink>
                ) : null}
                {workflow.capabilities.reuse_settings ? (
                  <ButtonLink
                    href={`/app/create/${workflow.id}?prompt=${encodeURIComponent(job.prompt)}`}
                    variant="ghost"
                    size="md"
                  >
                    <Icon name="reuse" size={15} />
                    Reuse Settings
                  </ButtonLink>
                ) : null}
                {workflow.capabilities.variation ? (
                  <ButtonLink
                    href={`/app/create/${workflow.id}?prompt=${encodeURIComponent(job.prompt)}`}
                    variant="ghost"
                    size="md"
                  >
                    <Icon name="copy" size={15} />
                    Variation
                  </ButtonLink>
                ) : null}
              </div>
            ) : null}
          </div>

          {/* ── Metadata ───────────────────────────────────────────────── */}
          <aside className="bg-zx-surface border-zx-border rounded-zx-lg h-fit border p-5">
            <h2 className="text-zx-text-muted m-0 mb-4 text-[11px] font-extrabold tracking-[0.11em] uppercase">
              Details
            </h2>
            <dl className="m-0 flex flex-col gap-4">
              <Detail label="Prompt" value={job.prompt || "—"} />
              <Detail label="Workflow" value={job.workflow_name} />
              {/* No duration parameter means the workflow set it from the
                  uploaded file — say so rather than showing a dash. */}
              <Detail
                label="Duration"
                value={
                  job.parameters?.duration
                    ? durationLabel(String(job.parameters.duration))
                    : "Matched to your file"
                }
              />
              {job.parameters?.aspect_ratio ? (
                <Detail label="Aspect ratio" value={String(job.parameters.aspect_ratio)} />
              ) : null}
              {job.parameters?.quality ? (
                <Detail label="Quality" value={String(job.parameters.quality)} />
              ) : null}
              <Detail label="Created" value={relativeTime(job.created_at)} />
              {job.attempt_count > 1 ? (
                <Detail label="Attempts" value={String(job.attempt_count)} />
              ) : null}
              <Detail label="Generation ID" value={job.id} />
            </dl>
          </aside>
        </div>
      )}
    </AppPage>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-zx-text-muted mb-1 text-[11.5px] font-bold">{label}</dt>
      <dd
        className="text-zx-text m-0 text-[13px] leading-[1.5] font-semibold break-words"
        suppressHydrationWarning
      >
        {value}
      </dd>
    </div>
  );
}
