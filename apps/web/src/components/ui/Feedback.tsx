import type { CSSProperties, ReactNode } from "react";
import { Icon, type IconName } from "./Icon";
import { cn } from "@/lib/cn";

/**
 * Status, empty and loading presentation — shared so every screen reads the
 * same way. Loading and empty states are part of what the client approves
 * (PRE-M1 directive: "loading states", "empty states").
 */

export type StatusTone = "running" | "success" | "error" | "muted";

const TONE_COLOR: Record<StatusTone, string> = {
  running: "bg-zx-accent",
  success: "bg-zx-success",
  error: "bg-zx-error",
  muted: "bg-zx-text-muted",
};

const TONE_TEXT: Record<StatusTone, string> = {
  running: "text-zx-primary-light",
  success: "text-zx-success",
  error: "text-zx-error",
  muted: "text-zx-text-muted",
};

/** Small coloured dot preceding a status label. Pulses while work is running. */
export function StatusDot({
  tone,
  className,
}: {
  tone: StatusTone;
  className?: string;
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "h-[6px] w-[6px] shrink-0 rounded-full",
        TONE_COLOR[tone],
        tone === "running" && "animate-zx-pulse",
        className,
      )}
    />
  );
}

export function StatusPill({
  tone,
  children,
}: {
  tone: StatusTone;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-[6px] rounded-md bg-[rgba(10,10,11,0.7)] px-[9px] py-[3px] text-[10.5px] font-extrabold backdrop-blur-[4px]",
        TONE_TEXT[tone],
      )}
    >
      <StatusDot tone={tone} />
      {children}
    </span>
  );
}

/** Consistent empty state for every list screen. */
export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: IconName;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="animate-zx-fade-up flex flex-col items-center px-6 py-16 text-center">
      <div className="bg-zx-surface-elevated border-zx-border text-zx-accent mb-5 flex h-14 w-14 items-center justify-center rounded-[16px] border">
        <Icon name={icon} size={22} />
      </div>
      <div className="text-zx-text mb-[7px] text-[16px] font-extrabold">
        {title}
      </div>
      <p className="text-zx-text-secondary max-w-[380px] text-[13.5px] leading-[1.55]">
        {description}
      </p>
      {action ? <div className="mt-5">{action}</div> : null}
    </div>
  );
}

/** Shimmer placeholder — the same treatment as the generation canvas. */
export function Skeleton({
  className,
  style,
}: {
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <div
      aria-hidden="true"
      style={style}
      className={cn(
        "animate-zx-shimmer bg-[linear-gradient(110deg,var(--zx-surface)_30%,#1E2418_50%,var(--zx-surface)_70%)] bg-[length:200%_100%]",
        className,
      )}
    />
  );
}
