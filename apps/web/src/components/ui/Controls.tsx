"use client";

import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

/**
 * Form controls shared by the settings panel and the app screens.
 * All sizing is taken from the approved design, including its off-scale values.
 */

/** The uppercase muted label above every settings group. */
export function SectionLabel({
  children,
  className,
  as: Tag = "div",
  ...rest
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "label" | "legend";
  /** Only meaningful with `as="label"`. */
  htmlFor?: string;
} & React.HTMLAttributes<HTMLElement>) {
  return (
    <Tag
      className={cn(
        "text-zx-text-muted mb-[10px] text-[11px] font-extrabold tracking-[0.11em] uppercase",
        className,
      )}
      {...rest}
    >
      {children}
    </Tag>
  );
}

/**
 * Selectable pill — durations, aspect ratios, filters, categories.
 * `aria-pressed` communicates selection to screen readers.
 */
export function OptionChip({
  selected,
  children,
  className,
  ...rest
}: {
  selected: boolean;
  children: ReactNode;
  className?: string;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      className={cn(
        "rounded-zx-sm cursor-pointer border transition-colors duration-150",
        selected
          ? "border-zx-border-active bg-zx-primary/16 text-zx-primary-light font-extrabold"
          : "border-zx-border bg-zx-surface text-zx-text-secondary hover:border-zx-border-active font-semibold",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

export interface SegmentedOption {
  value: string;
  label: string;
}

/**
 * Quality selector — a single track with the selection filled in, matching
 * the approved design's inset control.
 */
export function SegmentedControl({
  options,
  value,
  onChange,
  label,
}: {
  options: SegmentedOption[];
  value: string | null;
  onChange: (value: string) => void;
  label: string;
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className="bg-zx-surface border-zx-border flex gap-[3px] rounded-[10px] border p-[3px]"
    >
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={selected}
            onClick={() => onChange(option.value)}
            className={cn(
              "flex-1 cursor-pointer rounded-[7px] px-1 py-2 text-[12px] transition-colors duration-150",
              selected
                ? "bg-zx-primary/28 text-zx-text font-extrabold"
                : "text-zx-text-muted hover:text-zx-text-secondary font-semibold",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/** Labelled slider with a live value readout. */
export function RangeField({
  id,
  label,
  value,
  onChange,
  suffix = "%",
  min = 0,
  max = 100,
}: {
  id: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  suffix?: string;
  min?: number;
  max?: number;
}) {
  return (
    <div>
      <div className="text-zx-text-secondary mb-[7px] flex justify-between text-[12px] font-bold">
        <label htmlFor={id}>{label}</label>
        <span className="text-zx-primary-light">
          {value}
          {suffix}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="accent-zx-primary w-full cursor-pointer"
      />
    </div>
  );
}

/** Checkbox row — e.g. "Seed lock". */
export function ToggleField({
  id,
  label,
  checked,
  onChange,
}: {
  id: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label
      htmlFor={id}
      className="text-zx-text-secondary flex cursor-pointer items-center justify-between text-[12px] font-bold"
    >
      {label}
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="accent-zx-primary h-4 w-4 cursor-pointer"
      />
    </label>
  );
}

/** Search / text input styled to match the settings panel's textarea. */
export function TextField({
  className,
  ...rest
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "bg-zx-surface border-zx-border text-zx-text rounded-zx-md focus:border-zx-border-active w-full border px-[13px] py-[10px] text-[13px] outline-none transition-colors duration-150",
        className,
      )}
      {...rest}
    />
  );
}

/** Native select styled to match TextField — used where a choice list is too
 * long for chips (e.g. lyric languages). */
export function SelectField({
  className,
  children,
  ...rest
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        "bg-zx-surface border-zx-border text-zx-text rounded-zx-md focus:border-zx-border-active w-full cursor-pointer appearance-none border px-[13px] py-[10px] text-[13px] outline-none transition-colors duration-150",
        className,
      )}
      {...rest}
    >
      {children}
    </select>
  );
}
