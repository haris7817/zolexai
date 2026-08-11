"use client";

import { useState, useCallback, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import { BrandMark } from "./BrandMark";
import { NavLinks } from "./NavLinks";
import { UserCard } from "./UserCard";
import { ActiveJobsIndicator } from "./ActiveJobsIndicator";
import { Icon } from "@/components/ui/Icon";
import { DemoBadge } from "@/components/ui/DemoDisclosure";
import { useBodyScrollLock } from "@/hooks/useBodyScrollLock";
import { useEscapeKey } from "@/hooks/useEscapeKey";

/**
 * ===========================================================================
 * AppShell — sidebar + mobile header + navigation drawer
 * ===========================================================================
 *
 * Responsive modes (all CSS-driven, so nothing depends on measuring the window
 * during render — see hooks/useBreakpoint.ts for why that matters):
 *
 *   mobile   <768   top bar + slide-in drawer
 *   tablet   768+   64px icon rail, labels hidden
 *   laptop   1024+  200px sidebar with labels
 *   desktop  1440+  224px sidebar with labels
 */
export function AppShell({ children }: { children: ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const pathname = usePathname();

  const closeDrawer = useCallback(() => setDrawerOpen(false), []);

  useBodyScrollLock(drawerOpen);
  useEscapeKey(drawerOpen, closeDrawer);

  // Route change closes the drawer — otherwise it stays open over the new page.
  const handleNavigate = useCallback(() => setDrawerOpen(false), []);
  void pathname;

  return (
    <div className="bg-zx-bg text-zx-text flex min-h-screen flex-col tablet:h-screen tablet:overflow-hidden">
      {/* ── Mobile top bar ───────────────────────────────────────────── */}
      <header className="bg-zx-bg-alt border-zx-border sticky top-0 z-30 flex items-center justify-between gap-[10px] border-b px-4 py-3 tablet:hidden">
        <BrandMark href="/app" size="sm" />
        <div className="flex items-center gap-2">
          <ActiveJobsIndicator variant="header" />
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open navigation menu"
            aria-expanded={drawerOpen}
            className="bg-zx-surface border-zx-border text-zx-text rounded-zx-sm flex h-11 w-11 cursor-pointer items-center justify-center border"
          >
            <Icon name="menu" size={20} />
          </button>
        </div>
      </header>

      {/* ── Mobile navigation drawer ─────────────────────────────────── */}
      {drawerOpen ? (
        <>
          <div
            onClick={closeDrawer}
            aria-hidden="true"
            className="fixed inset-0 z-48 bg-black/55 tablet:hidden"
          />
          <aside
            role="dialog"
            aria-modal="true"
            aria-label="Navigation menu"
            className="bg-zx-bg-alt border-zx-border animate-zx-slide-in fixed inset-y-0 left-0 z-49 flex w-[282px] flex-col overflow-y-auto border-r px-[14px] py-[18px] shadow-[24px_0_60px_rgba(0,0,0,0.45)] tablet:hidden"
          >
            <div className="flex items-center justify-between px-[6px] pb-[18px]">
              <BrandMark href="/app" size="sm" />
              <button
                type="button"
                onClick={closeDrawer}
                aria-label="Close menu"
                className="bg-zx-surface border-zx-border text-zx-text-secondary rounded-zx-sm flex h-9 w-9 cursor-pointer items-center justify-center border"
              >
                <Icon name="close" size={16} />
              </button>
            </div>

            <NavLinks variant="drawer" onNavigate={handleNavigate} />
            <ActiveJobsIndicator />
            <UserCard variant="drawer" />
          </aside>
        </>
      ) : null}

      <div className="flex min-h-0 flex-1">
        {/* ── Sidebar: 64px rail → 200px → 224px ─────────────────────── */}
        <aside
          data-qa="sidebar"
          className="bg-zx-bg-alt border-zx-border hidden w-16 shrink-0 flex-col border-r px-[10px] py-5 tablet:flex laptop:w-[200px] laptop:px-3 desktop:w-[224px]"
        >
          <div className="flex justify-center pb-5 laptop:justify-start laptop:px-[10px]">
            <BrandMark href="/app" size="sm" showWordmark={false} />
            <span className="ml-[10px] hidden text-[16px] font-extrabold tracking-[-0.02em] laptop:inline">
              ZolexAI
            </span>
          </div>

          <NavLinks />
          <ActiveJobsIndicator />

          {/* Icon rail keeps a compact preview marker where the user card won't fit */}
          <div className="mt-3 flex justify-center laptop:hidden">
            <DemoBadge compact />
          </div>

          <UserCard />
        </aside>

        {/* ── Page content ───────────────────────────────────────────── */}
        {/* The ONE main landmark for every app page. Pages must not nest
            another <main> inside it — the workspace's canvas column is a
            <section> for exactly that reason. */}
        <main className="flex min-w-0 flex-1 flex-col tablet:min-h-0 tablet:overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  );
}
