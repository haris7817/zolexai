import { WORKFLOW_LIST } from "@/features/workflows/registry";
import type { IconName } from "@/components/ui/Icon";

/**
 * ONE navigation config for every navigation surface:
 *   · DesktopSidebar (224 / 200 / 64px icon rail)
 *   · MobileNavDrawer
 *   · Creator Dashboard sidebar
 *
 * The three source design files each hardcoded their own copy of this list,
 * which is exactly how they drifted ("Video Generator" vs "Text to Video").
 * Collapsing them here is required by architecture doc §40 and is the
 * acceptance criterion for M1.08 — "no duplicated hard-coded workflow pages".
 *
 * The CREATE section derives its tools from the workflow registry, so adding a
 * workflow in M2 updates navigation automatically and cannot fall out of sync.
 */

export interface NavItem {
  name: string;
  icon: IconName;
  href: string;
  /** True when the route also matches nested paths (e.g. /app/generations/123). */
  matchNested?: boolean;
}

export interface NavSection {
  label: string;
  items: NavItem[];
}

export const navigationSections: NavSection[] = [
  {
    label: "CREATE",
    items: [
      { name: "Home", icon: "home", href: "/app" },
      ...WORKFLOW_LIST.map(
        (workflow): NavItem => ({
          name: workflow.name,
          icon: workflow.icon,
          href: `/app/create/${workflow.id}`,
        }),
      ),
    ],
  },
  {
    label: "EXPLORE",
    items: [
      { name: "All Tools", icon: "grid", href: "/app/tools" },
      {
        name: "Generations",
        icon: "history",
        href: "/app/generations",
        matchNested: true,
      },
      { name: "Media Library", icon: "folder", href: "/app/media" },
    ],
  },
  {
    label: "ACCOUNT",
    items: [
      { name: "Subscription", icon: "card", href: "/app/subscription" },
      { name: "Settings", icon: "settings", href: "/app/settings" },
    ],
  },
];

/**
 * Active-route test. `/app` must match exactly, otherwise Home would light up
 * on every screen in the application.
 */
export function isNavItemActive(item: NavItem, pathname: string): boolean {
  if (item.href === "/app") return pathname === "/app";
  if (item.matchNested) return pathname.startsWith(item.href);
  return pathname === item.href;
}
