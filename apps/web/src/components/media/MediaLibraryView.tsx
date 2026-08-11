"use client";

import { useMemo, useRef, useState } from "react";
import { useQueryClient, useQuery, useInfiniteQuery } from "@tanstack/react-query";
import type { Asset, AssetKind } from "@zolexai/workflow-contracts";
import { AppPage, PageHeader } from "@/components/ui/PageHeader";
import { OptionChip, TextField } from "@/components/ui/Controls";
import { EmptyState, Skeleton } from "@/components/ui/Feedback";
import { Icon, type IconName } from "@/components/ui/Icon";
import { Button } from "@/components/ui/Button";
import { MediaPreview } from "./MediaPreview";
import { queryKeys } from "@/lib/query";
import { ApiError } from "@/lib/api/client";
import {
  fetchDownloadUrl,
  fetchMediaCounts,
  formatBytes,
  formatDuration,
  kindForFile,
  listMedia,
  uploadAsset,
} from "@/services/assets";
import { cn } from "@/lib/cn";

/**
 * Media Library — every uploaded and generated asset.
 *
 * Uploads go straight to object storage via a presigned PUT; nothing streams
 * through Next.js or the API. Listing is keyset-paginated and filtered
 * server-side, so the page cost does not grow with library size.
 */

const TABS: { value: AssetKind | "all"; label: string; icon: IconName }[] = [
  { value: "all", label: "All", icon: "folder" },
  { value: "video", label: "Videos", icon: "video" },
  { value: "image", label: "Images", icon: "picture" },
  { value: "audio", label: "Audio", icon: "audio" },
];

const SOURCES = [
  { value: null, label: "All sources" },
  { value: "generated" as const, label: "Generated" },
  { value: "upload" as const, label: "Uploaded" },
];

export function MediaLibraryView() {
  const [tab, setTab] = useState<AssetKind | "all">("all");
  const [sourceIndex, setSourceIndex] = useState(0);
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const source = SOURCES[sourceIndex].value;
  const filters = { kind: tab === "all" ? null : tab, source };

  const media = useInfiniteQuery({
    queryKey: queryKeys.media.list(filters),
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      listMedia({ limit: 24, cursor: pageParam, ...filters }, signal),
    getNextPageParam: (lastPage) => (lastPage.has_more ? lastPage.next_cursor : undefined),
  });

  const counts = useQuery({
    queryKey: queryKeys.media.counts,
    queryFn: ({ signal }) => fetchMediaCounts(signal),
  });

  const items = useMemo(
    () => media.data?.pages.flatMap((page) => page.items) ?? [],
    [media.data],
  );

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((item) => item.name.toLowerCase().includes(needle));
  }, [items, query]);

  return (
    <AppPage>
      <PageHeader
        title="Media Library"
        description="Everything you've uploaded or generated, ready to reuse in any workflow."
      />

      <UploadPanel />

      {/* ── Tabs ─────────────────────────────────────────────────────── */}
      <div
        role="tablist"
        aria-label="Media type"
        className="border-zx-border mb-4 flex gap-1 overflow-x-auto border-b"
      >
        {TABS.map((option) => {
          const active = option.value === tab;
          const count =
            option.value === "all" ? counts.data?.all : counts.data?.[option.value];
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
              {count !== undefined ? (
                <span className="text-zx-text-muted font-semibold">{count}</span>
              ) : null}
            </button>
          );
        })}
      </div>

      <div className="tablet:flex-row tablet:items-center mb-5 flex flex-col gap-3">
        <div className="relative flex-1">
          <span className="text-zx-text-muted pointer-events-none absolute top-1/2 left-3 -translate-y-1/2">
            <Icon name="search" size={15} />
          </span>
          <TextField
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search loaded media…"
            aria-label="Search media by name"
            className="pl-9"
          />
        </div>
        <div role="group" aria-label="Filter by source" className="flex gap-2">
          {SOURCES.map((option, index) => (
            <OptionChip
              key={option.label}
              selected={index === sourceIndex}
              onClick={() => setSourceIndex(index)}
              className="px-[14px] py-[9px] text-[12px]"
            >
              {option.label}
            </OptionChip>
          ))}
        </div>
      </div>

      {media.isPending ? (
        <div className="tablet:grid-cols-3 laptop:grid-cols-4 grid grid-cols-2 gap-[14px]">
          {Array.from({ length: 8 }).map((_, index) => (
            <Skeleton
              key={index}
              className="rounded-[14px]"
              style={{ aspectRatio: "16 / 13" }}
            />
          ))}
        </div>
      ) : media.isError ? (
        <EmptyState
          icon="alert"
          title="Couldn't load your media"
          description="The service may be temporarily unavailable. Try again in a moment."
          action={
            <Button variant="ghost" size="md" onClick={() => media.refetch()}>
              Try again
            </Button>
          }
        />
      ) : results.length > 0 ? (
        <>
          <div className="tablet:grid-cols-3 laptop:grid-cols-4 grid grid-cols-2 gap-[14px]">
            {results.map((item) => (
              <AssetCard
                key={item.id}
                item={item}
                selected={item.id === selectedId}
                onSelect={() => setSelectedId(item.id === selectedId ? null : item.id)}
              />
            ))}
          </div>

          {media.hasNextPage ? (
            <div className="mt-7 flex justify-center">
              <Button
                variant="ghost"
                size="md"
                onClick={() => media.fetchNextPage()}
                disabled={media.isFetchingNextPage}
              >
                {media.isFetchingNextPage ? "Loading…" : "Load more"}
              </Button>
            </div>
          ) : null}
        </>
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

/** Direct-to-storage upload, accepting any supported media type. */
function UploadPanel() {
  const queryClient = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setError(null);
    setBusy(true);
    try {
      for (const file of Array.from(files)) {
        const kind = kindForFile(file);
        if (!kind) {
          setError(`${file.name} is not a supported media type.`);
          continue;
        }
        await uploadAsset(file, kind);
      }
      await queryClient.invalidateQueries({ queryKey: queryKeys.media.all });
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause.message
          : "The upload could not be completed. Please try again.",
      );
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div className="mb-6">
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        disabled={busy}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          void handleFiles(event.dataTransfer.files);
        }}
        className="rounded-zx-md hover:border-zx-border-active hover:bg-zx-primary/4 w-full cursor-pointer border-[1.5px] border-dashed border-white/16 px-4 py-5 text-center transition-colors duration-150 disabled:cursor-wait"
      >
        <div className="text-zx-accent mb-[7px] flex justify-center">
          <Icon name="upload" size={20} />
        </div>
        <div className="text-zx-text text-[12.5px] font-bold">
          {busy ? "Uploading…" : "Drop images, video or audio here"}
        </div>
        <div className="text-zx-text-muted mt-[3px] text-[11.5px]">
          {busy ? "Sending straight to storage" : "or browse your files"}
        </div>
      </button>

      {error ? (
        <p role="alert" className="text-zx-error mt-2 text-[12px] font-semibold">
          {error}
        </p>
      ) : null}

      <input
        ref={fileRef}
        type="file"
        multiple
        accept="video/*,image/*,audio/*"
        onChange={(event) => void handleFiles(event.target.files)}
        className="hidden"
        aria-label="Upload media"
      />
    </div>
  );
}

function AssetCard({
  item,
  selected,
  onSelect,
}: {
  item: Asset;
  selected: boolean;
  onSelect: () => void;
}) {
  const [downloading, setDownloading] = useState(false);
  const kindIcon: IconName =
    item.kind === "audio" ? "audio" : item.kind === "image" ? "picture" : "video";

  const download = async () => {
    setDownloading(true);
    try {
      window.location.assign(await fetchDownloadUrl(item.id));
    } finally {
      setDownloading(false);
    }
  };

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
      >
        <MediaPreview
          url={item.url}
          kind={item.kind}
          aspectRatio="16 / 10"
          fallbackGradient="linear-gradient(140deg, #1C232A, #0B0E11)"
          className="pointer-events-none"
        />

        <span className="text-zx-text absolute top-2 left-2 flex items-center gap-[5px] rounded-md bg-[rgba(10,10,11,0.7)] px-[7px] py-[3px] text-[10px] font-bold backdrop-blur-[4px]">
          <Icon name={kindIcon} size={11} />
          {item.source === "generated" ? "Generated" : "Upload"}
        </span>

        {selected ? (
          <span className="bg-zx-primary text-zx-on-primary absolute top-2 right-2 flex h-5 w-5 items-center justify-center rounded-full">
            <Icon name="check" size={12} />
          </span>
        ) : null}

        {formatDuration(item.duration_seconds) ? (
          <span className="text-zx-text absolute right-2 bottom-2 rounded-md bg-[rgba(10,10,11,0.7)] px-[7px] py-[3px] text-[10px] font-bold">
            {formatDuration(item.duration_seconds)}
          </span>
        ) : null}
      </button>

      <div className="px-3 py-[10px]">
        <div className="text-zx-text truncate text-[12.5px] font-bold">{item.name}</div>
        <div className="text-zx-text-muted mt-[2px] text-[11px]">
          {item.width && item.height ? `${item.width} × ${item.height} · ` : ""}
          {formatBytes(item.size_bytes)}
        </div>

        <div className="mt-[10px] flex gap-[6px]">
          <Button
            variant="ghost"
            size="sm"
            onClick={download}
            disabled={downloading}
            className="flex-1 px-2 py-[6px] text-[11.5px]"
          >
            <Icon name="download" size={13} />
            {downloading ? "…" : "Save"}
          </Button>
        </div>
      </div>
    </div>
  );
}
