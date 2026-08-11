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

  /**
   * Downloads are a short-lived signed URL fetched on demand, not a link baked
   * into the page. Two reasons: a URL rendered at page load would already be
   * expired for a tab left open, and the bucket stays private so a copied link
   * cannot outlive its window.
   */
  const handleDownload = async () => {
    if (!output) return;
    setDownloading(true);
    try {
      const url = await fetchDownloadUrl(output.asset_id);
      window.location.assign(url);
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

      {capabilities.extend ? (
        <ButtonLink href="/app/create/extend-video" variant="ghost" size="md">
          <ActionIcon name="extend" />
          Extend
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
    </div>
  );
}

function ActionIcon({ name }: { name: IconName }) {
  return <Icon name={name} size={15} />;
}
