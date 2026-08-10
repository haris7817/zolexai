"use client";

import { useMemo, useState } from "react";
import { mockGenerations, type MockGeneration } from "@/mocks/generations";
import { WORKFLOW_LIST } from "@/features/workflows/registry";
import { GenerationCard } from "./GenerationCard";
import { AppPage, PageHeader } from "@/components/ui/PageHeader";
import { OptionChip, TextField } from "@/components/ui/Controls";
import { EmptyState } from "@/components/ui/Feedback";
import { Icon } from "@/components/ui/Icon";

/**
 * Generations — PREUI-07.
 *
 * ⚠️ MOCK: filtering runs client-side over `src/mocks/generations.ts`. Nothing
 * persists and there is no pagination. Real history arrives at M3.08.
 *
 * Deliberately includes a Failed row so the client can approve how an error
 * reads. Note the copy is friendly and generic — architecture rule #10 forbids
 * raw worker/model errors ever reaching a customer.
 */

const STATUSES = ["All", "Completed", "Generating", "Queued", "Failed"] as const;

export function GenerationsView() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<string>("All");
  const [workflowId, setWorkflowId] = useState<string>("all");

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return mockGenerations
      .filter((generation) => {
        const matchesStatus = status === "All" || generation.status === status;
        const matchesWorkflow =
          workflowId === "all" || generation.workflowId === workflowId;
        const matchesQuery =
          needle.length === 0 ||
          generation.prompt.toLowerCase().includes(needle);
        return matchesStatus && matchesWorkflow && matchesQuery;
      })
      .sort((a, b) => a.order - b.order);
  }, [query, status, workflowId]);

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
            placeholder="Search prompts…"
            aria-label="Search generations"
            className="pl-9"
          />
        </div>

        <div className="flex flex-wrap gap-2" role="group" aria-label="Filter by status">
          {STATUSES.map((option) => (
            <OptionChip
              key={option}
              selected={option === status}
              onClick={() => setStatus(option)}
              className="px-[14px] py-[8px] text-[12px]"
            >
              {option}
              {option !== "All" ? (
                <span className="text-zx-text-muted ml-[6px] font-semibold">
                  {countBy(option)}
                </span>
              ) : null}
            </OptionChip>
          ))}
        </div>

        <div className="flex flex-wrap gap-2" role="group" aria-label="Filter by workflow">
          <OptionChip
            selected={workflowId === "all"}
            onClick={() => setWorkflowId("all")}
            className="px-[14px] py-[8px] text-[12px]"
          >
            All workflows
          </OptionChip>
          {WORKFLOW_LIST.map((workflow) => (
            <OptionChip
              key={workflow.id}
              selected={workflowId === workflow.id}
              onClick={() => setWorkflowId(workflow.id)}
              className="inline-flex items-center gap-[6px] px-[14px] py-[8px] text-[12px]"
            >
              <Icon name={workflow.icon} size={13} />
              {workflow.name}
            </OptionChip>
          ))}
        </div>
      </div>

      <p className="text-zx-text-muted mb-4 text-[12px] font-bold">
        {results.length} {results.length === 1 ? "generation" : "generations"}
      </p>

      {results.length > 0 ? (
        <div className="grid grid-cols-1 gap-[14px] tablet:grid-cols-2 laptop:grid-cols-3 desktop:grid-cols-4">
          {results.map((generation) => (
            <GenerationCard key={generation.id} generation={generation} />
          ))}
        </div>
      ) : (
        <EmptyState
          icon="history"
          title="No generations match these filters"
          description="Try clearing the status or workflow filter, or search for a different prompt."
        />
      )}
    </AppPage>
  );
}

function countBy(status: string): number {
  return mockGenerations.filter(
    (generation: MockGeneration) => generation.status === status,
  ).length;
}
