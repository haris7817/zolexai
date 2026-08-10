"use client";

import { Button, ButtonLink } from "@/components/ui/Button";
import { Icon, type IconName } from "@/components/ui/Icon";
import type { WorkflowDefinition } from "@/features/workflows/types";

/**
 * Result actions, driven ENTIRELY by workflow capabilities.
 *
 * This is the mechanism architecture doc §9 and M2.20 require: "no irrelevant
 * action appears". A Music result has `extend: false`, so Extend is simply
 * never rendered — there is no branch on workflow id anywhere in this file.
 *
 * Worth demonstrating explicitly (guide §7 Step 6): switch to Music, generate,
 * and note that Extend is absent while Download / Reuse / Variation remain.
 */
export function ResultActions({
  workflow,
  onReuseSettings,
  onVariation,
}: {
  workflow: WorkflowDefinition;
  onReuseSettings: () => void;
  onVariation: () => void;
}) {
  const { capabilities } = workflow;

  return (
    <div className="mt-[18px] flex flex-wrap justify-center gap-[9px]">
      {capabilities.download ? (
        <Button variant="primary" size="md" onClick={noopDownload}>
          <ActionIcon name="download" />
          Download
        </Button>
      ) : null}

      {capabilities.extend ? (
        <ButtonLink href="/app/create/extend-video" variant="ghost" size="md">
          <ActionIcon name="extend" />
          Extend
        </ButtonLink>
      ) : null}

      {capabilities.reuseSettings ? (
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

/**
 * ⚠️ MOCK — there is no generated file to download. Real signed download URLs
 * arrive at M3.06. Kept as a no-op so the button is present for design review
 * without implying a working export.
 */
function noopDownload() {
  /* intentionally empty — see comment above */
}
