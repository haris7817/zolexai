"use client";

import { useMemo, useState } from "react";
import { WorkflowCard } from "./WorkflowCard";
import { AppPage, PageHeader } from "@/components/ui/PageHeader";
import { OptionChip, TextField } from "@/components/ui/Controls";
import { EmptyState } from "@/components/ui/Feedback";
import { Icon } from "@/components/ui/Icon";
import { useWorkflows } from "@/features/workflows/queries";

/**
 * All Tools.
 *
 * The catalogue comes from `GET /api/v1/workflows`. Search and category are
 * filtered client-side, which is correct here and only here: the whole
 * catalogue is six items served in one response, so a round trip per keystroke
 * would be pure latency for no benefit. Collections that can grow without
 * bound — generations, media — filter server-side instead.
 */

const CATEGORIES = [
  { value: "all", label: "All tools" },
  { value: "video", label: "Video" },
  { value: "audio", label: "Audio" },
] as const;

export function AllToolsView() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<string>("all");
  const { workflows } = useWorkflows();

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return workflows.filter((workflow) => {
      const matchesCategory = category === "all" || workflow.category === category;
      const matchesQuery =
        needle.length === 0 ||
        workflow.name.toLowerCase().includes(needle) ||
        workflow.description.toLowerCase().includes(needle) ||
        workflow.marketing_description.toLowerCase().includes(needle);
      return matchesCategory && matchesQuery;
    });
  }, [workflows, query, category]);

  return (
    <AppPage>
      <PageHeader
        title="All Tools"
        description="Every ZolexAI creation workflow, organised by what it produces."
      />

      <div className="tablet:flex-row tablet:items-center mb-6 flex flex-col gap-3">
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
        <div className="tablet:grid-cols-2 desktop:grid-cols-3 grid grid-cols-1 gap-5">
          {results.map((workflow) => (
            <WorkflowCard key={workflow.id} workflow={workflow} tone="detailed" />
          ))}
        </div>
      ) : (
        <EmptyState
          icon="search"
          title="No tools match that search"
          description="Try a different term, or clear the category filter to see every workflow."
        />
      )}
    </AppPage>
  );
}
