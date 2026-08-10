"use client";

import { useState, useCallback } from "react";
import Link from "next/link";
import { BrandMark } from "@/components/navigation/BrandMark";
import { ButtonLink } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import { useBodyScrollLock } from "@/hooks/useBodyScrollLock";
import { useEscapeKey } from "@/hooks/useEscapeKey";

const LINKS = [
  { label: "Features", href: "#tools" },
  { label: "Tools", href: "#tools" },
  { label: "How it works", href: "#how" },
  { label: "Pricing", href: "#pricing" },
];

/**
 * Landing header.
 *
 * The original design had no mobile treatment at all — a single flex row with
 * 32px gaps that overflows below ~900px. It collapses to a sheet here (ADR 0001).
 */
export function MarketingNav() {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);

  useBodyScrollLock(open);
  useEscapeKey(open, close);

  return (
    <nav className="relative z-10 mx-auto flex max-w-[1280px] items-center justify-between px-5 py-5 tablet:px-8 laptop:px-12">
      <BrandMark href="/" size="lg" />

      <div className="hidden items-center gap-8 laptop:flex">
        {LINKS.map((link) => (
          <Link
            key={link.label}
            href={link.href}
            className="text-zx-text-secondary hover:text-zx-text text-[14px] font-semibold"
          >
            {link.label}
          </Link>
        ))}
        <Link
          href="/login"
          className="text-zx-text hover:text-zx-text text-[14px] font-semibold"
        >
          Sign In
        </Link>
        <ButtonLink href="/app" variant="primary" size="md" className="px-[22px] py-[10px] text-[14px]">
          Start Creating
        </ButtonLink>
      </div>

      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label="Open menu"
        aria-expanded={open}
        className="bg-zx-surface border-zx-border text-zx-text rounded-zx-sm flex h-11 w-11 cursor-pointer items-center justify-center border laptop:hidden"
      >
        <Icon name="menu" size={20} />
      </button>

      {open ? (
        <>
          <div
            onClick={close}
            aria-hidden="true"
            className="fixed inset-0 z-40 bg-black/60 laptop:hidden"
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Menu"
            className="bg-zx-bg-alt border-zx-border animate-zx-fade-up fixed inset-x-0 top-0 z-50 flex flex-col gap-1 border-b px-5 pt-5 pb-6 laptop:hidden"
          >
            <div className="mb-4 flex items-center justify-between">
              <BrandMark href="/" size="sm" />
              <button
                type="button"
                onClick={close}
                aria-label="Close menu"
                className="bg-zx-surface border-zx-border text-zx-text-secondary rounded-zx-sm flex h-9 w-9 cursor-pointer items-center justify-center border"
              >
                <Icon name="close" size={16} />
              </button>
            </div>

            {LINKS.map((link) => (
              <Link
                key={link.label}
                href={link.href}
                onClick={close}
                className="text-zx-text-secondary hover:text-zx-text rounded-zx-sm px-2 py-3 text-[15px] font-semibold"
              >
                {link.label}
              </Link>
            ))}
            <Link
              href="/login"
              onClick={close}
              className="text-zx-text rounded-zx-sm px-2 py-3 text-[15px] font-semibold"
            >
              Sign In
            </Link>
            <ButtonLink
              href="/app"
              variant="primary"
              size="lg"
              fullWidth
              className="mt-3"
              onClick={close}
            >
              Start Creating
            </ButtonLink>
          </div>
        </>
      ) : null}
    </nav>
  );
}
