import type { ReactNode } from "react";

/**
 * Consistent heading for every app screen, so Generations, Media, Subscription
 * and Settings introduce themselves the same way the Workspace does.
 */
export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
      <div>
        <h1 className="text-zx-text m-0 text-[22px] font-extrabold tracking-[-0.02em] laptop:text-[26px]">
          {title}
        </h1>
        <p className="text-zx-text-secondary mt-[3px] mb-0 text-[13.5px]">
          {description}
        </p>
      </div>
      {actions ? <div className="flex gap-2">{actions}</div> : null}
    </header>
  );
}

/** Shared page frame — scrolls internally on tablet+, with the page on mobile. */
export function AppPage({ children }: { children: ReactNode }) {
  return (
    <div className="mx-auto w-full max-w-[1180px] px-4 pt-6 pb-16 tablet:min-h-0 tablet:overflow-y-auto tablet:px-8 laptop:px-10 laptop:pt-8">
      {children}
    </div>
  );
}
