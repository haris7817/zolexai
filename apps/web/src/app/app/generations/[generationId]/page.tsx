import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { mockGenerations } from "@/mocks/generations";
import { WORKFLOWS } from "@/features/workflows/registry";
import { AppPage } from "@/components/ui/PageHeader";
import { Button, ButtonLink } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import { StatusPill } from "@/components/ui/Feedback";
import { statusTone } from "@/components/generation/GenerationCard";
import { DemoSimulationNote } from "@/components/ui/DemoDisclosure";

export function generateStaticParams() {
  return mockGenerations.map((generation) => ({
    generationId: generation.id,
  }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ generationId: string }>;
}): Promise<Metadata> {
  const { generationId } = await params;
  const generation = mockGenerations.find((item) => item.id === generationId);
  return { title: generation ? "Generation" : "Not found" };
}

/**
 * Generation detail — part of PREUI-07.
 *
 * ⚠️ MOCK. Real detail, lineage and downloads arrive at M3.09.
 *
 * Result actions come from the workflow's capabilities, exactly as in the
 * workspace — so an audio generation shows no Extend here either.
 */
export default async function GenerationDetailPage({
  params,
}: {
  params: Promise<{ generationId: string }>;
}) {
  const { generationId } = await params;
  const generation = mockGenerations.find((item) => item.id === generationId);
  if (!generation) notFound();

  const workflow = WORKFLOWS[generation.workflowId];
  const isComplete = generation.status === "Completed";
  const isAudio = generation.outputType === "audio";

  return (
    <AppPage>
      <Link
        href="/app/generations"
        className="text-zx-text-secondary hover:text-zx-text mb-5 inline-flex items-center gap-[6px] text-[13px] font-bold"
      >
        <Icon name="chevronLeft" size={15} />
        Back to generations
      </Link>

      <div className="grid grid-cols-1 gap-6 laptop:grid-cols-[1fr_300px]">
        <div>
          <div
            className="border-zx-border relative flex items-center justify-center overflow-hidden rounded-[14px] border"
            style={{
              background: generation.thumb,
              aspectRatio: isAudio ? "16 / 7" : "16 / 9",
            }}
          >
            <span className="absolute top-3 left-3">
              <StatusPill tone={statusTone(generation.status)}>
                {generation.status.toUpperCase()}
              </StatusPill>
            </span>

            {isComplete ? (
              <button
                type="button"
                aria-label={isAudio ? "Play track" : "Play video"}
                className="flex h-[60px] w-[60px] cursor-pointer items-center justify-center rounded-full border border-white/25 bg-white/10 backdrop-blur-[6px] transition-colors duration-150 hover:bg-white/18"
              >
                <span
                  aria-hidden="true"
                  className="ml-[4px] h-0 w-0 border-y-[10px] border-l-[16px] border-y-transparent border-l-white"
                />
              </button>
            ) : null}

            <span className="text-zx-text absolute right-3 bottom-3 rounded-md bg-[rgba(13,12,19,0.7)] px-[9px] py-1 text-[11px] font-bold">
              {generation.duration}
            </span>
          </div>

          {generation.status === "Failed" && generation.errorMessage ? (
            <div className="border-zx-error/40 bg-zx-error/8 rounded-zx-md mt-4 flex items-start gap-3 border p-4">
              <span className="text-zx-error mt-[1px]">
                <Icon name="alert" size={16} />
              </span>
              <div>
                <div className="text-zx-text mb-1 text-[13px] font-bold">
                  Generation failed
                </div>
                <p className="text-zx-text-secondary m-0 text-[12.5px] leading-[1.55]">
                  {generation.errorMessage}
                </p>
              </div>
            </div>
          ) : null}

          {isComplete ? (
            <div className="mt-[18px] flex flex-wrap gap-[9px]">
              {workflow.capabilities.download ? (
                <Button variant="primary" size="md">
                  <Icon name="download" size={15} />
                  Download
                </Button>
              ) : null}
              {workflow.capabilities.extend ? (
                <ButtonLink href="/app/create/extend-video" variant="ghost" size="md">
                  <Icon name="extend" size={15} />
                  Extend
                </ButtonLink>
              ) : null}
              {workflow.capabilities.reuseSettings ? (
                <ButtonLink
                  href={`/app/create/${workflow.id}`}
                  variant="ghost"
                  size="md"
                >
                  <Icon name="reuse" size={15} />
                  Reuse Settings
                </ButtonLink>
              ) : null}
              {workflow.capabilities.variation ? (
                <ButtonLink
                  href={`/app/create/${workflow.id}`}
                  variant="ghost"
                  size="md"
                >
                  <Icon name="copy" size={15} />
                  Variation
                </ButtonLink>
              ) : null}
            </div>
          ) : null}

          <DemoSimulationNote className="mt-5 text-left" />
        </div>

        {/* ── Metadata ───────────────────────────────────────────────── */}
        <aside className="bg-zx-surface border-zx-border rounded-zx-lg h-fit border p-5">
          <h2 className="text-zx-text-muted m-0 mb-4 text-[11px] font-extrabold tracking-[0.11em] uppercase">
            Details
          </h2>
          <dl className="m-0 flex flex-col gap-4">
            <Detail label="Prompt" value={generation.prompt} />
            <Detail label="Workflow" value={generation.workflowName} />
            <Detail label="Duration" value={generation.duration} />
            {generation.aspect ? (
              <Detail label="Aspect ratio" value={generation.aspect} />
            ) : null}
            {generation.quality ? (
              <Detail label="Quality" value={generation.quality} />
            ) : null}
            <Detail label="Created" value={generation.createdLabel} />
            <Detail label="Generation ID" value={generation.id} />
          </dl>
        </aside>
      </div>
    </AppPage>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-zx-text-muted mb-1 text-[11.5px] font-bold">
        {label}
      </dt>
      <dd className="text-zx-text m-0 text-[13px] leading-[1.5] font-semibold">
        {value}
      </dd>
    </div>
  );
}
