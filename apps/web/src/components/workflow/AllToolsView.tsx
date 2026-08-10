"use client";

import { useMemo, useState } from "react";
import { WORKFLOW_LIST } from "@/features/workflows/registry";
import { WorkflowCard } from "./WorkflowCard";
import { AppPage, PageHeader } from "@/components/ui/PageHeader";
import { OptionChip, TextField } from "@/components/ui/Controls";
import { EmptyState } from "@/components/ui/Feedback";
import { Icon } from "@/components/ui/Icon";

/**
 * All Tools — PREUI-06.
 *
 * ⚠️ MOCK filtering: search and category run client-side over the registry.
 * There is no backend. Real workflow metadata arrives from
 * GET /api/v1/workflows at M1.15.
 *
 * Renders the six frozen-scope workflows and nothing else — same registry as
 * the sidebar, Dashboard and Landing, so the four can never disagree.
 */

const CATEGORIES = [
  { value: "all", label: "All tools" },
  { value: "video", label: "Video" },
  { value: "audio", label: "Audio" },
] as const;

export function AllToolsView() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<string>("all");

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return WORKFLOW_LIST.filter((workflow) => {
      const matchesCategory =
        category === "all" || workflow.category === category;
      const matchesQuery =
        needle.length === 0 ||
        workflow.name.toLowerCase().includes(needle) ||
        workflow.description.toLowerCase().includes(needle) ||
        workflow.marketingDescription.toLowerCase().includes(needle);
      return matchesCategory && matchesQuery;
    });
  }, [query, category]);

  return (
    <AppPage>
      <PageHeader
        title="All Tools"
        description="Every ZolexAI creation workflow, organised by what it produces."
      />

      <div className="mb-6 flex flex-col gap-3 tablet:flex-row tablet:items-center">
        <div className="relative flex-1">
          <span className="text-zx-text-muted pointer-events-none absolute top-1/2 left-3 -translate-y-1/2">
            <Icon name="search" size={15} />
          </span>
          <TextField
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search tools…"
            aria-label="Search tools"
            className="pl-9"
          />
        </div>

        <div role="group" aria-label="Filter by category" className="flex gap-2">
          {CATEGORIES.map((option) => (
            <OptionChip
              key={option.value}
              selected={option.value === category}
              onClick={() => setCategory(option.value)}
              className="px-4 py-[9px] text-[12px]"
            >
              {option.label}
            </OptionChip>
          ))}
        </div>
      </div>

      {results.length > 0 ? (
        <div className="grid grid-cols-1 gap-5 tablet:grid-cols-2 desktop:grid-cols-3">
          {results.map((workflow) => (
            <WorkflowCard key={workflow.id} workflow={workflow} tone="detailed" />
          ))}
        </div>
      ) : (
        <EmptyState
          icon="search"
          title="No tools match that search"
          description="Try a different term, or clear the category filter to see all six workflows."
        />
      )}
    </AppPage>
  );
}
