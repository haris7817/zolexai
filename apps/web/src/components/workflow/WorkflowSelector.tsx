"use client";

import Link from "next/link";
import { Icon } from "@/components/ui/Icon";
import { useWorkflows } from "@/features/workflows/queries";
import { cn } from "@/lib/cn";

/**
 * The 2×3 workflow grid in the settings panel.
 *
 * Rendered from the catalogue the API served, so it can never disagree with the
 * sidebar, All Tools, the Dashboard or the Landing page. Selecting a workflow
 * is real navigation (`/app/create/{id}`), which keeps the workspace URL
 * shareable and gives every tool its own metadata title.
 */
export function WorkflowSelector({
  activeId,
  onSelect,
}: {
  activeId: string;
  onSelect?: () => void;
}) {
  const { workflows } = useWorkflows();

  return (
    <div className="mb-6 grid grid-cols-2 gap-2">
      {workflows.map((workflow) => {
        const selected = workflow.id === activeId;
        return (
          <Link
            key={workflow.id}
            href={`/app/create/${workflow.id}`}
            onClick={onSelect}
            aria-current={selected ? "page" : undefined}
            className={cn(
              "hover:border-zx-border-active flex cursor-pointer flex-col items-center gap-[6px] rounded-[10px] border px-2 py-[10px] text-[12px] transition-colors duration-150",
              selected
                ? "border-zx-border-active bg-zx-primary/16 text-zx-primary-light hover:text-zx-primary-light font-extrabold"
                : "border-zx-border bg-zx-surface text-zx-text-secondary hover:text-zx-text-secondary font-semibold",
            )}
          >
            <Icon name={workflow.ui.icon} size={17} />
            <span className="text-center leading-tight">{workflow.name}</span>
          </Link>
        );
      })}
    </div>
  );
}
