"use client";

import Link from "next/link";
import { GenerationCard } from "@/components/generation/GenerationCard";
import { EmptyState, Skeleton } from "@/components/ui/Feedback";
import { Icon } from "@/components/ui/Icon";
import { useGenerationHistory } from "@/features/generation/queries";

/**
 * The dashboard's "Recent generations" strip.
 *
 * A client island inside a server-rendered page: the page itself is static
 * chrome and the workflow grid (both known at build time), and only this
 * section needs live per-user data.
 */
export function RecentGenerations() {
  const history = useGenerationHistory({});
  const items = (history.data?.pages[0]?.items ?? []).slice(0, 4);

  if (history.isPending) {
    return (
      <div className="tablet:grid-cols-2 desktop:grid-cols-4 mb-12 grid grid-cols-1 gap-[14px]">
        {Array.from({ length: 4 }).map((_, index) => (
          <Skeleton key={index} className="rounded-[14px]" style={{ aspectRatio: "16 / 13" }} />
        ))}
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="border-zx-border rounded-zx-lg mb-12 border">
        <EmptyState
          icon="sparkles"
          title="Nothing generated yet"
          description="Pick a tool above, describe what you want, and your results will collect here."
          action={
            <Link
              href="/app/create/text-to-video"
              className="text-zx-primary-light hover:text-zx-text inline-flex items-center gap-1 text-[13px] font-bold"
            >
              Start with Text to Video
              <Icon name="arrowRight" size={13} />
            </Link>
          }
        />
      </div>
    );
  }

  return (
    <div className="tablet:grid-cols-2 desktop:grid-cols-4 mb-12 grid grid-cols-1 gap-[14px]">
      {items.map((generation) => (
        <GenerationCard key={generation.id} generation={generation} />
      ))}
    </div>
  );
}
