import Link from "next/link";
import { brand } from "@/config/brand";
import { cn } from "@/lib/cn";

/**
 * The ZolexAI gradient "Z" tile plus wordmark.
 *
 * All three source design files drew this independently at slightly different
 * sizes; it is one component now so the mark can never drift between the
 * landing page, the app shell and the mobile drawer.
 */
export function BrandMark({
  href = "/app",
  size = "md",
  showWordmark = true,
  className,
}: {
  href?: string;
  size?: "sm" | "md" | "lg";
  showWordmark?: boolean;
  className?: string;
}) {
  const tile = {
    sm: "h-[28px] w-[28px] rounded-lg text-[14px]",
    md: "h-[30px] w-[30px] rounded-[9px] text-[15px]",
    lg: "h-[32px] w-[32px] rounded-[9px] text-[16px]",
  }[size];

  const word = {
    sm: "text-[16px]",
    md: "text-[17px]",
    lg: "text-[19px]",
  }[size];

  return (
    <Link
      href={href}
      aria-label={`${brand.name} home`}
      className={cn(
        "text-zx-text hover:text-zx-text flex items-center gap-[10px]",
        className,
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "flex shrink-0 items-center justify-center bg-[image:var(--zx-gradient-primary)] font-extrabold text-white",
          tile,
        )}
      >
        {brand.shortName}
      </span>
      {showWordmark ? (
        <span className={cn("font-extrabold tracking-[-0.02em]", word)}>
          {brand.name}
        </span>
      ) : null}
    </Link>
  );
}
