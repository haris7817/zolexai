import Link from "next/link";
import { mockUser } from "@/mocks/user";
import { Icon } from "@/components/ui/Icon";
import { DemoBadge } from "@/components/ui/DemoDisclosure";
import { cn } from "@/lib/cn";

/**
 * Account block at the foot of the sidebar and drawer.
 *
 * ⚠️ MOCK user — there is no authentication in this phase (M3.02). "Log out"
 * returns to the landing page so the demo never dead-ends.
 */
export function UserCard({
  variant = "sidebar",
}: {
  variant?: "sidebar" | "drawer";
}) {
  const isDrawer = variant === "drawer";

  return (
    <div
      className={cn(
        "border-zx-border mt-[14px] border-t pt-3",
        // Hidden on the 64px icon rail — no room for a name and plan.
        !isDrawer && "hidden laptop:block",
      )}
    >
      <DemoBadge className="mb-3 ml-[6px]" />

      <div className="flex items-center gap-[10px] pl-[6px]">
        <span
          aria-hidden="true"
          className="flex h-[30px] w-[30px] shrink-0 items-center justify-center rounded-full bg-[linear-gradient(135deg,var(--zx-accent),#6E8F05)] text-[11px] font-extrabold text-zx-on-primary"
        >
          {mockUser.initials}
        </span>

        <div className="min-w-0 flex-1">
          <div className="text-zx-text truncate text-[12.5px] font-bold">
            {mockUser.name}
          </div>
          <div className="text-zx-text-muted text-[11px]">
            {mockUser.plan} plan
          </div>
        </div>

        <Link
          href="/"
          aria-label="Log out"
          title="Log out"
          className="text-zx-text-muted hover:text-zx-text flex rounded-md p-[6px] transition-colors duration-150"
        >
          <Icon name="logout" size={15} />
        </Link>
      </div>
    </div>
  );
}
