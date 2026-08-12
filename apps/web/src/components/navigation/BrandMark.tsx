import Image from "next/image";
import Link from "next/link";
import { brand } from "@/config/brand";
import { cn } from "@/lib/cn";

/**
 * The real ZolexAI monogram plus text wordmark.
 *
 * One component feeds the landing nav, the app shell, the mobile drawer, the
 * auth screens and the footer, so the mark can never drift between surfaces —
 * which is also why swapping the placeholder gradient tile for the client's
 * actual logo (CR-003) was a change to this file alone.
 *
 * The wordmark stays as text: the logo's own chrome lettering is beautiful at
 * hero scale and illegible at 17px. The image files live in `public/brand/`,
 * cut from the client-provided master (transparent background, so they sit on
 * any surface the black/neon theme uses).
 */

/** Trimmed monogram crop — 897×783 in the master, hence the ratio. */
const MARK_RATIO = 897 / 783;

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
  const height = { sm: 28, md: 30, lg: 34 }[size];

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
      <Image
        src="/brand/logo-mark.png"
        // The link's aria-label already names the destination; a second
        // "ZolexAI" here would be read twice by a screen reader.
        alt=""
        width={Math.round(height * MARK_RATIO)}
        height={height}
        // A 16KB static PNG needs no optimizer pass, and skipping it keeps
        // behaviour identical between dev, `next start` and any static deploy.
        unoptimized
        className="shrink-0"
      />
      {showWordmark ? (
        <span className={cn("font-extrabold tracking-[-0.02em]", word)}>
          {brand.name}
        </span>
      ) : null}
    </Link>
  );
}
