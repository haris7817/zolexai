"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { Asset, WorkflowInput } from "@zolexai/workflow-contracts";
import { ApiError } from "@/lib/api/client";
import { queryKeys } from "@/lib/query";
import { fetchAsset, formatBytes, uploadAsset } from "@/services/assets";
import { Icon } from "./Icon";
import { cn } from "@/lib/cn";

/**
 * Media input — a real direct-to-storage upload.
 *
 *   Browser ──presigned PUT──▶ Object storage
 *
 * The file never passes through Next.js or FastAPI (directive §13). This
 * component asks the API for a signed URL, uploads straight to storage, then
 * confirms so the API can verify what actually landed.
 *
 * Accepted types and the size ceiling come from the workflow definition, so the
 * control enforces exactly what the API will accept — and a workflow added in
 * M2 gets correct validation with no change here.
 */
export function Dropzone({
  input,
  value,
  onChange,
  className,
}: {
  input: WorkflowInput;
  value: string | null;
  onChange: (assetId: string | null) => void;
  className?: string;
}) {
  const queryClient = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const previewUrlRef = useRef<string | null>(null);

  const [uploaded, setUploaded] = useState<Asset | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewPlaying, setPreviewPlaying] = useState(false);

  const maxBytes = input.max_size_mb * 1024 * 1024;

  /**
   * A value can arrive without this component having uploaded anything —
   * Extend hands over an existing generation as its source. Resolve it so the
   * control shows what it is about to use instead of an empty upload box that
   * contradicts a form the user cannot see is already valid.
   */
  const needsLookup = Boolean(value) && uploaded?.id !== value;
  const { data: resolved } = useQuery({
    queryKey: queryKeys.media.detail(value ?? ""),
    queryFn: ({ signal }) => fetchAsset(value as string, signal),
    enabled: needsLookup,
    staleTime: Infinity,
  });

  const asset = uploaded?.id === value ? uploaded : (resolved ?? null);

  const resetPreview = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setPreviewPlaying(false);

    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);
  }, []);

  useEffect(() => {
    if (!previewUrl) return;
    const audio = audioRef.current;

    return () => {
      audio?.pause();
      if (previewUrlRef.current === previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrlRef.current = null;
      }
    };
  }, [previewUrl]);

  // A parent form can replace or clear the selected asset without using this
  // component's remove button. A local preview only belongs to the exact asset
  // created from its browser File, so dispose it as soon as that asset changes.
  useEffect(() => {
    if (previewUrl && uploaded?.id !== value) resetPreview();
  }, [previewUrl, resetPreview, uploaded?.id, value]);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);

      // Checked before requesting a URL: the API would reject these too, but
      // catching them here saves a round trip and reads instantly.
      if (input.accept.length && !input.accept.includes(file.type)) {
        setError(`That file type is not supported for ${input.label.toLowerCase()}.`);
        return;
      }
      if (file.size > maxBytes) {
        setError(`That file is larger than ${input.max_size_mb} MB.`);
        return;
      }

      resetPreview();
      setUploading(true);
      try {
        const asset = await uploadAsset(file, input.kind);
        if (input.kind === "audio") {
          const objectUrl = URL.createObjectURL(file);
          previewUrlRef.current = objectUrl;
          setPreviewUrl(objectUrl);
        }
        setUploaded(asset);
        onChange(asset.id);
        // The new file belongs in the media library immediately.
        await queryClient.invalidateQueries({ queryKey: queryKeys.media.all });
      } catch (cause) {
        setError(
          cause instanceof ApiError
            ? cause.message
            : "The upload could not be completed. Please try again.",
        );
        onChange(null);
      } finally {
        setUploading(false);
      }
    },
    [input, maxBytes, onChange, queryClient, resetPreview],
  );

  const clear = () => {
    resetPreview();
    setUploaded(null);
    setError(null);
    onChange(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  if (value && asset) {
    return (
      <div
        className={cn(
          "rounded-zx-md border-zx-border bg-zx-surface flex items-center gap-3 border p-3",
          className,
        )}
      >
        <span className="text-zx-primary-light shrink-0">
          <Icon name="check" size={16} />
        </span>
        <span className="min-w-0 flex-1">
          <span className="text-zx-text block truncate text-[12.5px] font-bold">
            {asset.name}
          </span>
          <span className="text-zx-text-muted text-[11px]">
            {formatBytes(asset.size_bytes)}
          </span>
        </span>
        {input.kind === "audio" && previewUrl && uploaded?.id === value ? (
          <>
            <button
              type="button"
              onClick={() => {
                const audio = audioRef.current;
                if (!audio) return;
                if (audio.paused) {
                  void audio.play().catch(() => {
                    setPreviewPlaying(false);
                    setError("This audio file could not be previewed in your browser.");
                  });
                } else {
                  audio.pause();
                }
              }}
              aria-label={`${previewPlaying ? "Pause" : "Play"} preview of ${asset.name}`}
              className={cn(
                "border-zx-primary/35 bg-zx-primary/10 text-zx-primary-light hover:border-zx-primary/65 hover:bg-zx-primary/18",
                "flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full border transition-colors duration-150",
              )}
            >
              <Icon name={previewPlaying ? "pause" : "play"} size={14} />
            </button>
            <audio
              ref={audioRef}
              src={previewUrl}
              preload="metadata"
              onPlay={() => setPreviewPlaying(true)}
              onPause={() => setPreviewPlaying(false)}
              onEnded={(event) => {
                event.currentTarget.currentTime = 0;
                setPreviewPlaying(false);
              }}
            />
          </>
        ) : null}
        <button
          type="button"
          onClick={clear}
          aria-label={`Remove ${asset.name}`}
          className="text-zx-text-muted hover:text-zx-text shrink-0 cursor-pointer transition-colors duration-150"
        >
          <Icon name="close" size={15} />
        </button>
      </div>
    );
  }

  return (
    <div className={className}>
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        disabled={uploading}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files?.[0];
          if (file) void handleFile(file);
        }}
        className={cn(
          "rounded-zx-md w-full cursor-pointer border-[1.5px] border-dashed px-4 py-5 text-center transition-colors duration-150",
          "hover:border-zx-border-active hover:bg-zx-primary/4 disabled:cursor-wait",
          dragging ? "border-zx-border-active bg-zx-primary/6" : "border-white/16",
          error && "border-zx-error/50",
        )}
      >
        <div className="text-zx-accent mb-[7px] flex justify-center">
          <Icon name="upload" size={20} />
        </div>
        <div className="text-zx-text text-[12.5px] font-bold">
          {uploading ? "Uploading…" : `Drop ${input.drop_hint} here`}
        </div>
        <div className="text-zx-text-muted mt-[3px] text-[11.5px]">
          {uploading ? "Sending straight to storage" : `or browse · up to ${input.max_size_mb} MB`}
        </div>
      </button>

      {input.help ? (
        <p className="text-zx-text-muted mt-[6px] text-[11.5px] leading-[1.45]">
          {input.help}
        </p>
      ) : null}

      {error ? (
        <p role="alert" className="text-zx-error mt-[6px] text-[11.5px] font-semibold">
          {error}
        </p>
      ) : null}

      <input
        ref={fileRef}
        type="file"
        accept={input.accept.join(",")}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void handleFile(file);
        }}
        className="hidden"
        aria-label={input.label}
      />
    </div>
  );
}
