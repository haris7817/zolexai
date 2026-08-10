import type { Metadata } from "next";
import Link from "next/link";
import { WORKFLOW_LIST } from "@/features/workflows/registry";
import { mockGenerations, mockDrafts } from "@/mocks/generations";
import { mockUser } from "@/mocks/user";
import { QuickStart } from "@/components/dashboard/QuickStart";
import { WorkflowCard } from "@/components/workflow/WorkflowCard";
import { GenerationCard } from "@/components/generation/GenerationCard";
import { Icon } from "@/components/ui/Icon";

export const metadata: Metadata = { title: "Home" };

/**
 * Creator Dashboard — PREUI-04.
 *
 * Deliberately creator-focused, NOT an analytics dashboard: no charts, no KPI
 * tiles, no usage graphs. The first thing on screen is a way to start creating.
 *
 * Retro-fitted from the original design onto the unified token system (it was
 * drawn against the older #0B0A14 / #110E1E palette with Unicode glyph icons)
 * and given the responsive behaviour it never had — the original was a fixed
 * 240px sidebar with hard 3- and 4-column grids. See ADR 0001.
 */
export default function DashboardPage() {
  const recents = mockGenerations.slice(0, 4);
  const firstName = mockUser.name.split(" ")[0];

  return (
    <div className="mx-auto w-full max-w-[1180px] px-4 pt-8 pb-16 tablet:px-8 laptop:px-12 laptop:pt-11">
      <header className="mb-7">
        <h1 className="text-zx-text m-0 mb-[6px] text-[26px] font-extrabold tracking-[-0.03em] laptop:text-[34px]">
          What are you creating today?
        </h1>
        <p className="text-zx-text-secondary m-0 text-[14px] laptop:text-[15px]">
          Welcome back, {firstName}. Start with a prompt, or jump into any tool
          below.
        </p>
      </header>

      <QuickStart />

      <SectionHeader title="Tools" actionLabel="All tools" href="/app/tools" />
      <div className="mb-12 grid grid-cols-2 gap-[14px] tablet:grid-cols-3 laptop:grid-cols-[repeat(auto-fill,minmax(170px,1fr))]">
        {WORKFLOW_LIST.map((workflow) => (
          <WorkflowCard key={workflow.id} workflow={workflow} />
        ))}
      </div>

      <SectionHeader title="Continue creating" />
      <div className="mb-12 grid grid-cols-1 gap-[14px] tablet:grid-cols-2 laptop:grid-cols-3">
        {mockDrafts.map((draft) => (
          <Link
            key={draft.id}
            href="/app/create/text-to-video"
            className="bg-zx-surface border-zx-border hover:border-zx-border-active flex items-center gap-[14px] rounded-[14px] border p-[14px] transition-colors duration-150"
          >
            <span
              aria-hidden="true"
              className="relative h-12 w-[72px] shrink-0 overflow-hidden rounded-[9px]"
              style={{ background: draft.thumb }}
            >
              <span
                className="bg-zx-accent absolute bottom-0 left-0 h-[3px]"
                style={{ width: draft.progress }}
              />
            </span>
            <span className="min-w-0">
              <span className="text-zx-text block truncate text-[13.5px] font-bold">
                {draft.title}
              </span>
              <span className="text-zx-text-muted mt-[3px] block text-[12px]">
                {draft.workflowName} · {draft.editedLabel}
              </span>
            </span>
          </Link>
        ))}
      </div>

      <SectionHeader
        title="Recent generations"
        actionLabel="View all"
        href="/app/generations"
      />
      <div className="mb-12 grid grid-cols-1 gap-[14px] tablet:grid-cols-2 desktop:grid-cols-4">
        {recents.map((generation) => (
          <GenerationCard key={generation.id} generation={generation} />
        ))}
      </div>

      <h2 className="text-zx-text m-0 mb-[18px] text-[19px] font-extrabold tracking-[-0.02em]">
        Recommended for you
      </h2>
      <div className="grid grid-cols-1 gap-[14px] laptop:grid-cols-2">
        <RecommendationCard
          icon="extend"
          title="Extend your latest video"
          description="Turn your 15-second commercial into a full 60-second cut."
          href="/app/create/extend-video"
        />
        <RecommendationCard
          icon="clapper"
          title="Try Music Video"
          description="Pair your synthwave track with visuals in one generation."
          href="/app/create/music-video"
        />
      </div>
    </div>
  );
}

function SectionHeader({
  title,
  actionLabel,
  href,
}: {
  title: string;
  actionLabel?: string;
  href?: string;
}) {
  return (
    <div className="mb-[18px] flex items-baseline justify-between gap-4">
      <h2 className="text-zx-text m-0 text-[19px] font-extrabold tracking-[-0.02em]">
        {title}
      </h2>
      {actionLabel && href ? (
        <Link
          href={href}
          className="text-zx-accent hover:text-zx-primary-light inline-flex items-center gap-1 text-[13px] font-bold"
        >
          {actionLabel}
          <Icon name="arrowRight" size={13} />
        </Link>
      ) : null}
    </div>
  );
}

function RecommendationCard({
  icon,
  title,
  description,
  href,
}: {
  icon: "extend" | "clapper";
  title: string;
  description: string;
  href: string;
}) {
  return (
    <Link
      href={href}
      className="rounded-zx-lg border-zx-border-active hover:border-zx-border-active flex items-center gap-[18px] border bg-[linear-gradient(120deg,rgba(109,61,245,0.14),rgba(23,22,31,0.6))] p-5 transition-colors duration-150"
    >
      <span
        aria-hidden="true"
        className="flex h-[46px] w-[46px] shrink-0 items-center justify-center rounded-[13px] bg-[image:var(--zx-gradient-primary)] text-white"
      >
        <Icon name={icon} size={20} />
      </span>
      <span>
        <span className="text-zx-text block text-[15px] font-extrabold">
          {title}
        </span>
        <span className="text-zx-text-secondary mt-[3px] block text-[13px] leading-[1.45]">
          {description}
        </span>
      </span>
    </Link>
  );
}
