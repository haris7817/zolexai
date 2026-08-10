import Link from "next/link";
import type { ComponentProps, ReactNode } from "react";
import { cn } from "@/lib/cn";

/**
 * ZolexAI button hierarchy, taken from the approved design.
 *
 *   primary  gradient fill + CTA shadow — one per view, the Generate/Start action
 *   ghost    surface + border           — secondary actions (History, Share, Extend)
 *   subtle   transparent                — tertiary/inline (Advanced settings)
 *
 * Sizes match the design's real values, which are off Tailwind's scale by
 * design (12.5px text, 9px/15px padding). Those are preserved exactly.
 */

export type ButtonVariant = "primary" | "ghost" | "subtle";
export type ButtonSize = "sm" | "md" | "lg";

const VARIANT: Record<ButtonVariant, string> = {
  primary:
    "bg-[image:var(--zx-gradient-primary)] text-white font-extrabold shadow-zx-cta hover:brightness-110",
  ghost:
    "bg-zx-surface border border-zx-border text-zx-text-secondary font-bold hover:text-zx-text hover:border-white/16",
  subtle:
    "bg-transparent text-zx-text-secondary font-bold hover:text-zx-text",
};

const SIZE: Record<ButtonSize, string> = {
  sm: "text-[12.5px] px-[14px] py-[8px] gap-[7px] rounded-zx-sm",
  md: "text-[12.5px] px-[15px] py-[9px] gap-[7px] rounded-zx-sm",
  // Touch-friendly: 48px min height satisfies the mobile QA requirement.
  lg: "text-[15px] px-[26px] py-[15px] gap-[8px] rounded-zx-md min-h-[48px]",
};

const BASE =
  "inline-flex items-center justify-center whitespace-nowrap cursor-pointer " +
  "transition-[background-color,border-color,color,filter] duration-150 ease-out " +
  "disabled:cursor-not-allowed disabled:opacity-45";

interface CommonProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  fullWidth?: boolean;
  className?: string;
  children: ReactNode;
}

export function Button({
  variant = "ghost",
  size = "md",
  fullWidth,
  className,
  ...rest
}: CommonProps & ComponentProps<"button">) {
  return (
    <button
      className={cn(
        BASE,
        VARIANT[variant],
        SIZE[size],
        fullWidth && "w-full",
        className,
      )}
      {...rest}
    />
  );
}

/** Same visual vocabulary, rendered as a link for real navigation. */
export function ButtonLink({
  variant = "ghost",
  size = "md",
  fullWidth,
  className,
  ...rest
}: CommonProps & ComponentProps<typeof Link>) {
  return (
    <Link
      className={cn(
        BASE,
        VARIANT[variant],
        SIZE[size],
        fullWidth && "w-full",
        // Links inherit the anchor colour rule in tokens.css; override it.
        "hover:no-underline",
        variant === "primary" && "text-white hover:text-white",
        className,
      )}
      {...rest}
    />
  );
}
