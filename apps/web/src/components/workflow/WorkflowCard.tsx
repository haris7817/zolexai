import Link from "next/link";
import { Icon } from "@/components/ui/Icon";
import type { WorkflowDefinition } from "@/features/workflows/types";
import { cn } from "@/lib/cn";

/**
 * Tool card — shared by the Creator Dashboard, All Tools and the Landing page.
 *
 * One component reading one registry is why those three surfaces can never
 * disagree about which tools exist. `tone` only changes copy density:
 *
 *   compact    dashboard grid      — short description
 *   detailed   All Tools / Landing — full marketing description + thumbnail
 */
export function WorkflowCard({
  workflow,
  tone = "compact",
  href,
}: {
  workflow: WorkflowDefinition;
  tone?: "compact" | "detailed";
  href?: string;
}) {
  const target = href ?? `/app/create/${workflow.id}`;

  if (tone === "detailed") {
    return (
      <Link
        href={target}
        className="bg-zx-surface border-zx-border rounded-zx-lg hover:border-zx-border-active block overflow-hidden border transition-[transform,border-color] duration-200 hover:-translate-y-[3px]"
      >
        <div
          className="relative h-[130px]"
          style={{ background: workflow.thumb }}
        >
          <span className="border-zx-border bg-zx-surface-elevated text-zx-primary-light absolute bottom-[-18px] left-5 flex h-10 w-10 items-center justify-center rounded-[11px] border">
            <Icon name={workflow.icon} size={18} />
          </span>
        </div>
        <div className="px-5 pt-[30px] pb-[22px]">
          <div className="text-zx-text mb-[6px] text-[16.5px] font-extrabold">
            {workflow.name}
          </div>
          <p className="text-zx-text-secondary text-[14px] leading-[1.5]">
            {workflow.marketingDescription}
          </p>
        </div>
      </Link>
    );
  }

  return (
    <Link
      href={target}
      className={cn(
        "bg-zx-surface border-zx-border hover:border-zx-border-active block rounded-[14px] border px-4 py-[18px] transition-[transform,border-color] duration-150 hover:-translate-y-[2px]",
      )}
    >
      <span
        className="mb-[14px] flex h-[38px] w-[38px] items-center justify-center rounded-[11px] text-[#E4DFFF]"
        style={{ background: workflow.thumb }}
      >
        <Icon name={workflow.icon} size={17} />
      </span>
      <div className="text-zx-text mb-1 text-[14.5px] font-extrabold">
        {workflow.name}
      </div>
      <div className="text-zx-text-muted text-[12.5px] leading-[1.45]">
        {workflow.shortDescription}
      </div>
    </Link>
  );
}
