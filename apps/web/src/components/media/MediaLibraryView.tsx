"use client";

import { useMemo, useState } from "react";
import { mockMedia, type MediaKind, type MockMediaItem } from "@/mocks/media";
import { AppPage, PageHeader } from "@/components/ui/PageHeader";
import { OptionChip, TextField } from "@/components/ui/Controls";
import { EmptyState } from "@/components/ui/Feedback";
import { Dropzone } from "@/components/ui/Dropzone";
import { Icon, type IconName } from "@/components/ui/Icon";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/cn";

/**
 * Media Library — PREUI-08.
 *
 * ⚠️ MOCK: tabs, source filter and search run client-side over
 * `src/mocks/media.ts`. The dropzone accepts nothing — direct-to-storage signed
 * uploads arrive at M3.05, and the library itself at M3.07.
 *
 * Selection is included because choosing media is how the library feeds the
 * workflows that need input (Image to Video, Video to Video, Extend, Music Video).
 */

const TABS: { value: MediaKind | "all"; label: string; icon: IconName }[] = [
  { value: "all", label: "All", icon: "folder" },
  { value: "video", label: "Videos", icon: "video" },
  { value: "image", label: "Images", icon: "picture" },
  { value: "audio", label: "Audio", icon: "audio" },
];

const SOURCES = [
  { value: "all", label: "All sources" },
  { value: "generated", label: "Generated" },
  { value: "upload", label: "Uploaded" },
] as const;

export function MediaLibraryView() {
  const [tab, setTab] = useState<MediaKind | "all">("all");
  const [source, setSource] = useState<string>("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return mockMedia
      .filter((item) => {
        const matchesTab = tab === "all" || item.kind === tab;
        const matchesSource = source === "all" || item.source === source;
        const matchesQuery =
          needle.length === 0 || item.name.toLowerCase().includes(needle);
        return matchesTab && matchesSource && matchesQuery;
      })
      .sort((a, b) => a.order - b.order);
  }, [tab, source, query]);

  return (
    <AppPage>
      <PageHeader
        title="Media Library"
        description="Everything you've uploaded or generated, ready to reuse in any workflow."
      />

      <Dropzone kind="images, video or audio" className="mb-6" />

      {/* ── Tabs ─────────────────────────────────────────────────────── */}
      <div
        role="tablist"
        aria-label="Media type"
        className="border-zx-border mb-4 flex gap-1 overflow-x-auto border-b"
      >
        {TABS.map((option) => {
          const active = option.value === tab;
          return (
            <button
              key={option.value}
              role="tab"
              type="button"
              aria-selected={active}
              onClick={() => setTab(option.value)}
              className={cn(
                "-mb-px flex cursor-pointer items-center gap-[7px] border-b-2 px-4 py-[10px] text-[13px] whitespace-nowrap transition-colors duration-150",
                active
                  ? "border-zx-accent text-zx-text font-extrabold"
                  : "text-zx-text-muted hover:text-zx-text-secondary border-transparent font-semibold",
              )}
            >
              <Icon name={option.icon} size={14} />
              {option.label}
              <span className="text-zx-text-muted font-semibold">
                {option.value === "all"
                  ? mockMedia.length
                  : mockMedia.filter((item) => item.kind === option.value).length}
              </span>
            </button>
          );
        })}
      </div>

      <div className="mb-5 flex flex-col gap-3 tablet:flex-row tablet:items-center">
        <div className="relative flex-1">
          <span className="text-zx-text-muted pointer-events-none absolute top-1/2 left-3 -translate-y-1/2">
            <Icon name="search" size={15} />
          </span>
          <TextField
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search media…"
            aria-label="Search media"
            className="pl-9"
          />
        </div>
        <div role="group" aria-label="Filter by source" className="flex gap-2">
          {SOURCES.map((option) => (
            <OptionChip
              key={option.value}
              selected={option.value === source}
              onClick={() => setSource(option.value)}
              className="px-[14px] py-[9px] text-[12px]"
            >
              {option.label}
            </OptionChip>
          ))}
        </div>
      </div>

      {results.length > 0 ? (
        <div className="grid grid-cols-2 gap-[14px] tablet:grid-cols-3 laptop:grid-cols-4">
          {results.map((item) => (
            <AssetCard
              key={item.id}
              item={item}
              selected={item.id === selectedId}
              onSelect={() =>
                setSelectedId(item.id === selectedId ? null : item.id)
              }
            />
          ))}
        </div>
      ) : (
        <EmptyState
          icon="folder"
          title="Nothing here yet"
          description="Upload media or generate something new — results are added to your library automatically."
        />
      )}
    </AppPage>
  );
}

function AssetCard({
  item,
  selected,
  onSelect,
}: {
  item: MockMediaItem;
  selected: boolean;
  onSelect: () => void;
}) {
  const kindIcon: IconName =
    item.kind === "audio" ? "audio" : item.kind === "image" ? "picture" : "video";

  return (
    <div
      className={cn(
        "bg-zx-surface overflow-hidden rounded-[14px] border transition-colors duration-150",
        selected
          ? "border-zx-border-active"
          : "border-zx-border hover:border-zx-border-active",
      )}
    >
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={selected}
        aria-label={`Select ${item.name}`}
        className="relative block w-full cursor-pointer"
        style={{ background: item.thumb, aspectRatio: "16 / 10" }}
      >
        <span className="text-zx-text absolute top-2 left-2 flex items-center gap-[5px] rounded-md bg-[rgba(10,10,11,0.7)] px-[7px] py-[3px] text-[10px] font-bold backdrop-blur-[4px]">
          <Icon name={kindIcon} size={11} />
          {item.source === "generated" ? "Generated" : "Upload"}
        </span>

        {selected ? (
          <span className="bg-zx-primary absolute top-2 right-2 flex h-5 w-5 items-center justify-center rounded-full text-zx-on-primary">
            <Icon name="check" size={12} />
          </span>
        ) : null}

        {item.duration ? (
          <span className="text-zx-text absolute right-2 bottom-2 rounded-md bg-[rgba(10,10,11,0.7)] px-[7px] py-[3px] text-[10px] font-bold">
            {item.duration}
          </span>
        ) : null}
      </button>

      <div className="px-3 py-[10px]">
        <div className="text-zx-text truncate text-[12.5px] font-bold">
          {item.name}
        </div>
        <div className="text-zx-text-muted mt-[2px] text-[11px]">
          {item.dimensions ?? item.duration} · {item.size}
        </div>

        <div className="mt-[10px] flex gap-[6px]">
          <Button variant="ghost" size="sm" className="flex-1 px-2 py-[6px] text-[11.5px]">
            <Icon name="download" size={13} />
            Save
          </Button>
          <Button
            variant="ghost"
            size="sm"
            aria-label={`More actions for ${item.name}`}
            className="px-2 py-[6px]"
          >
            <Icon name="more" size={13} />
          </Button>
        </div>
      </div>
    </div>
  );
}
