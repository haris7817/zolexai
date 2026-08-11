"use client";

import { useMemo, useState } from "react";
import type { JobStatus } from "@zolexai/workflow-contracts";
import { GenerationCard } from "./GenerationCard";
import { AppPage, PageHeader } from "@/components/ui/PageHeader";
import { OptionChip, TextField } from "@/components/ui/Controls";
import { EmptyState, Skeleton } from "@/components/ui/Feedback";
import { Icon } from "@/components/ui/Icon";
import { Button } from "@/components/ui/Button";
import { useGenerationHistory } from "@/features/generation/queries";
import { useWorkflows } from "@/features/workflows/queries";

/**
 * Generation history.
 *
 * Status and workflow filters are applied by the API (indexed, so they stay
 * fast on a large table), and pages are fetched with the API's opaque keyset
 * cursor — never an offset, and never "load everything then filter".
 *
 * The prompt search box is the one client-side filter, and deliberately so: it
 * narrows the pages already loaded rather than pretending to search all
 * history. Full-text search over prompts is a server capability that does not
 * exist yet; adding it here would silently return partial results. The label
 * says "in loaded results" so it does not overpromise.
 */

const STATUS_FILTERS: { label: string; value: JobStatus[] | null }[] = [
  { label: "All", value: null },
  { label: "Completed", value: ["completed"] },
  {
    label: "Running",
    value: ["queued", "assigned", "preparing", "generating", "post_processing", "uploading"],
  },
  { label: "Failed", value: ["failed", "cancelled"] },
];

export function GenerationsView() {
  const [query, setQuery] = useState("");
  const [statusIndex, setStatusIndex] = useState(0);
  const [workflowId, setWorkflowId] = useState<string | null>(null);

  const { workflows } = useWorkflows();
  const history = useGenerationHistory({
    status: STATUS_FILTERS[statusIndex].value ?? undefined,
    workflowId,
  });

  const items = useMemo(
    () => history.data?.pages.flatMap((page) => page.items) ?? [],
    [history.data],
  );

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((job) => job.prompt.toLowerCase().includes(needle));
  }, [items, query]);

  return (
    <AppPage>
      <PageHeader
        title="Generations"
        description="Everything you've created, with status, settings and results."
      />

      {/* ── Filters ──────────────────────────────────────────────────── */}
      <div className="mb-5 flex flex-col gap-3">
        <div className="relative">
          <span className="text-zx-text-muted pointer-events-none absolute top-1/2 left-3 -translate-y-1/2">
            <Icon name="search" size={15} />
          </span>
          <TextField
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search prompts in loaded results…"
            aria-label="Search loaded generations by prompt"
            className="pl-9"
          />
        </div>

        <div className="flex flex-wrap gap-2" role="group" aria-label="Filter by status">
          {STATUS_FILTERS.map((option, index) => (
            <OptionChip
              key={option.label}
              selected={index === statusIndex}
              onClick={() => setStatusIndex(index)}
              className="px-[14px] py-[8px] text-[12px]"
            >
              {option.label}
            </OptionChip>
          ))}
        </div>

        <div className="flex flex-wrap gap-2" role="group" aria-label="Filter by workflow">
          <OptionChip
            selected={workflowId === null}
            onClick={() => setWorkflowId(null)}
            className="px-[14px] py-[8px] text-[12px]"
          >
            All workflows
          </OptionChip>
          {workflows.map((workflow) => (
            <OptionChip
              key={workflow.id}
              selected={workflowId === workflow.id}
              onClick={() => setWorkflowId(workflow.id)}
              className="inline-flex items-center gap-[6px] px-[14px] py-[8px] text-[12px]"
            >
              <Icon name={workflow.ui.icon} size={13} />
              {workflow.name}
            </OptionChip>
          ))}
        </div>
      </div>

      {history.isPending ? (
        <div className="tablet:grid-cols-2 laptop:grid-cols-3 desktop:grid-cols-4 grid grid-cols-1 gap-[14px]">
          {Array.from({ length: 8 }).map((_, index) => (
            <Skeleton key={index} className="rounded-[14px]" style={{ aspectRatio: "16 / 13" }} />
          ))}
        </div>
      ) : history.isError ? (
        <EmptyState
          icon="alert"
          title="Couldn't load your generations"
          description="The service may be temporarily unavailable. Try again in a moment."
          action={
            <Button variant="ghost" size="md" onClick={() => history.refetch()}>
              Try again
            </Button>
          }
        />
      ) : results.length > 0 ? (
        <>
          <p className="text-zx-text-muted mb-4 text-[12px] font-bold">
            {results.length} {results.length === 1 ? "generation" : "generations"}
            {history.hasNextPage ? " loaded" : ""}
          </p>

          <div className="tablet:grid-cols-2 laptop:grid-cols-3 desktop:grid-cols-4 grid grid-cols-1 gap-[14px]">
            {results.map((generation) => (
              <GenerationCard key={generation.id} generation={generation} />
            ))}
          </div>

          {history.hasNextPage ? (
            <div className="mt-7 flex justify-center">
              <Button
                variant="ghost"
                size="md"
                onClick={() => history.fetchNextPage()}
                disabled={history.isFetchingNextPage}
              >
                {history.isFetchingNextPage ? "Loading…" : "Load more"}
              </Button>
            </div>
          ) : null}
        </>
      ) : (
        <EmptyState
          icon="history"
          title={items.length ? "No generations match these filters" : "Nothing generated yet"}
          description={
            items.length
              ? "Try clearing the status or workflow filter, or search for a different prompt."
              : "Pick a tool, describe what you want, and your results will collect here."
          }
        />
      )}
    </AppPage>
  );
}
