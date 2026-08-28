"use client";

import { useState } from "react";
import type { GenerationJob, Workflow } from "@zolexai/workflow-contracts";
import { Button, ButtonLink } from "@/components/ui/Button";
import { Icon, type IconName } from "@/components/ui/Icon";
import { primaryOutput } from "@/services/generations";
import { fetchDownloadUrl } from "@/services/assets";

/**
 * Result actions, driven ENTIRELY by workflow capabilities.
 *
 * A Music result has `extend: false`, so Extend is simply never rendered —
 * there is no branch on workflow id anywhere in this file. Adding a workflow in
 * M2 gets correct actions from its definition alone.
 */
export function ResultActions({
  job,
  workflow,
  onReuseSettings,
  onVariation,
}: {
  job: GenerationJob;
  workflow: Workflow;
  onReuseSettings: () => void;
  onVariation: () => void;
}) {
  const { capabilities } = workflow;
  const output = primaryOutput(job);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  /**
   * Downloads are a short-lived signed URL fetched on demand, not a link baked
   * into the page. Two reasons: a URL rendered at page load would already be
   * expired for a tab left open, and the bucket stays private so a copied link
   * cannot outlive its window.
   */
  const handleDownload = async () => {
    if (!output) return;
    setDownloading(true);
    setDownloadError(null);
    try {
      const url = await fetchDownloadUrl(output.asset_id);
      window.location.assign(url);
    } catch {
      // A failed signing request used to be swallowed here: the button said
      // "Preparing…", went back to "Download", and nothing happened — which is
      // indistinguishable from a browser that blocked the navigation, and is
      // what a customer reports as "I can't download anything" (28 Aug 2026).
      // Whatever the cause, saying so beats a button that does nothing twice.
      setDownloadError("That download could not be prepared. Please try again.");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="mt-[18px] flex flex-wrap justify-center gap-[9px]">
      {capabilities.download && output ? (
        <Button variant="primary" size="md" onClick={handleDownload} disabled={downloading}>
          <ActionIcon name="download" />
          {downloading ? "Preparing…" : "Download"}
        </Button>
      ) : null}

      {/* The result travels with the link as the extension's source, so the
          user is not asked to download their own generation and upload it
          back. Without an output there is nothing to extend from, so the
          action is not offered. */}
      {capabilities.extend && output ? (
        <ButtonLink
          href={`/app/create/extend-video?source=${output.asset_id}`}
          variant="ghost"
          size="md"
        >
          <ActionIcon name="extend" />
          Extend
        </ButtonLink>
      ) : null}

      {/* A finished TRACK is a valid source for the audio-driven video
          workflow, and the hand-off is the same shape as Extend's: the asset
          travels with the link, nothing is re-uploaded. Gated on the output's
          KIND — what the file is — mirroring how Extend gates on capability,
          so a video or image result never grows this button. */}
      {output?.kind === "audio" ? (
        <ButtonLink
          href={`/app/create/music-video?source=${output.asset_id}`}
          variant="ghost"
          size="md"
        >
          <ActionIcon name="clapper" />
          Generate Music Video
        </ButtonLink>
      ) : null}

      {capabilities.reuse_settings ? (
        <Button variant="ghost" size="md" onClick={onReuseSettings}>
          <ActionIcon name="reuse" />
          Reuse Settings
        </Button>
      ) : null}

      {capabilities.variation ? (
        <Button variant="ghost" size="md" onClick={onVariation}>
          <ActionIcon name="copy" />
          Variation
        </Button>
      ) : null}

      {downloadError ? (
        <p
          role="alert"
          className="text-zx-error w-full text-center text-[11.5px] font-semibold"
        >
          {downloadError}
        </p>
      ) : null}
    </div>
  );
}

function ActionIcon({ name }: { name: IconName }) {
  return <Icon name={name} size={15} />;
}
