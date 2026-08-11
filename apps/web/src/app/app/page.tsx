import type { Metadata } from "next";
import Link from "next/link";
import { loadWorkflowCatalog } from "@/features/workflows/catalog.server";
import { mockUser } from "@/mocks/user";
import { QuickStart } from "@/components/dashboard/QuickStart";
import { RecentGenerations } from "@/components/dashboard/RecentGenerations";
import { WorkflowCard } from "@/components/workflow/WorkflowCard";
import { Icon } from "@/components/ui/Icon";

export const metadata: Metadata = { title: "Home" };

/**
 * Creator Dashboard — PREUI-04.
 *
 * Deliberately creator-focused, NOT an analytics dashboard: no charts, no KPI
 * tiles, no usage graphs. The first thing on screen is a way to start creating.
 *
 * Retro-fitted from the original design onto the unified token system and
 * given the responsive behaviour it never had. See ADR 0001.
 *
 * The tool grid is server-rendered from the workflow definitions; recent
 * generations are a client island reading the API. The "Continue creating"
 * drafts row from the PRE-M1 mock is gone: drafts are not a concept the
 * platform has, and leaving a fabricated one in place would have promised a
 * feature no milestone builds.
 */
export default async function DashboardPage() {
  const workflows = await loadWorkflowCatalog();
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
        {workflows.map((workflow) => (
          <WorkflowCard key={workflow.id} workflow={workflow} />
        ))}
      </div>

      <SectionHeader
        title="Recent generations"
        actionLabel="View all"
        href="/app/generations"
      />
      <RecentGenerations />

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
      className="rounded-zx-lg border-zx-border-active hover:border-zx-border-active flex items-center gap-[18px] border bg-[linear-gradient(120deg,rgba(190,242,8,0.14),rgba(21,21,24,0.6))] p-5 transition-colors duration-150"
    >
      <span
        aria-hidden="true"
        className="flex h-[46px] w-[46px] shrink-0 items-center justify-center rounded-[13px] bg-[image:var(--zx-gradient-primary)] text-zx-on-primary"
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
