"use client";

import { useCallback, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { Asset, WorkflowInput } from "@zolexai/workflow-contracts";
import { ApiError } from "@/lib/api/client";
import { queryKeys } from "@/lib/query";
import { formatBytes, uploadAsset } from "@/services/assets";
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

  const [asset, setAsset] = useState<Asset | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const maxBytes = input.max_size_mb * 1024 * 1024;

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

      setUploading(true);
      try {
        const uploaded = await uploadAsset(file, input.kind);
        setAsset(uploaded);
        onChange(uploaded.id);
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
    [input, maxBytes, onChange, queryClient],
  );

  const clear = () => {
    setAsset(null);
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
