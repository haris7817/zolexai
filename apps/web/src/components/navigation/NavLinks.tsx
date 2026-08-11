"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  buildNavigationSections,
  isNavItemActive,
  type NavItem,
} from "@/config/navigation";
import { useWorkflows } from "@/features/workflows/queries";
import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

/**
 * The navigation list, rendered from the ONE config in config/navigation.ts.
 *
 * Shared by the desktop sidebar and the mobile drawer — the two surfaces differ
 * only in sizing, which is why `variant` is the sole difference below. The
 * source designs each hardcoded their own copy of this list, which is how
 * "Video Generator" and "Text to Video" drifted apart.
 */
export function NavLinks({
  variant = "sidebar",
  onNavigate,
}: {
  variant?: "sidebar" | "drawer";
  onNavigate?: () => void;
}) {
  const pathname = usePathname();
  const isDrawer = variant === "drawer";

  // Seeded on the server from the YAML catalogue, so this renders complete on
  // first paint rather than as an empty rail.
  const { workflows } = useWorkflows();
  const navigationSections = buildNavigationSections(workflows);

  return (
    <nav
      aria-label="Main navigation"
      className="flex min-w-0 flex-1 flex-col gap-[18px] overflow-x-hidden overflow-y-auto"
    >
      {navigationSections.map((section) => (
        <div key={section.label} className="flex flex-col gap-[1px]">
          <div
            className={cn(
              "text-zx-text-muted px-[10px] pb-[7px] text-[10.5px] font-extrabold tracking-[0.11em]",
              // The 64px icon rail (tablet) has no room for section labels.
              !isDrawer && "hidden laptop:block",
            )}
          >
            {section.label}
          </div>

          {section.items.map((item) => (
            <NavLinkItem
              key={item.href}
              item={item}
              active={isNavItemActive(item, pathname)}
              isDrawer={isDrawer}
              onNavigate={onNavigate}
            />
          ))}
        </div>
      ))}
    </nav>
  );
}

function NavLinkItem({
  item,
  active,
  isDrawer,
  onNavigate,
}: {
  item: NavItem;
  active: boolean;
  isDrawer: boolean;
  onNavigate?: () => void;
}) {
  return (
    <Link
      href={item.href}
      title={item.name}
      aria-current={active ? "page" : undefined}
      onClick={onNavigate}
      className={cn(
        "rounded-zx-sm flex items-center gap-[10px] font-semibold transition-colors duration-150",
        isDrawer
          ? "px-[10px] py-[11px] text-[14px]"
          : "px-[10px] py-[7px] text-[13px] justify-center laptop:justify-start",
        active
          ? "bg-zx-primary/10 text-zx-text shadow-[inset_2px_0_0_var(--zx-accent)]"
          : "text-zx-text-secondary hover:bg-zx-surface-hover hover:text-zx-text",
      )}
    >
      <span
        className={cn(
          "flex items-center",
          active ? "text-zx-accent" : "text-zx-text-muted",
        )}
      >
        <Icon name={item.icon} size={16} />
      </span>
      <span className={cn(!isDrawer && "hidden laptop:inline")}>
        {item.name}
      </span>
    </Link>
  );
}
