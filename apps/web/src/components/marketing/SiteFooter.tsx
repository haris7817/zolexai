import Link from "next/link";
import { brand } from "@/config/brand";
import { BrandMark } from "@/components/navigation/BrandMark";
import { DemoFooterNote } from "@/components/ui/DemoDisclosure";

/**
 * Landing footer.
 *
 * The original design was a hard 5-column grid (`2fr 1fr 1fr 1fr 1fr`) with no
 * responsive rule — it collapses to 2 columns on tablet and 1 on mobile here.
 *
 * Footer links are intentionally non-navigating: those pages are not part of
 * the current scope, so they are rendered as plain text rather than links that
 * 404. See the demo checklist note on dead ends.
 */

const COLUMNS = [
  { title: "PRODUCT", items: ["Tools", "Pricing", "Changelog"] },
  { title: "COMPANY", items: ["About", "Blog", "Careers"] },
  { title: "SUPPORT", items: ["Help Center", "Contact", "Status"] },
  { title: "LEGAL", items: ["Terms", "Privacy", "Cookies"] },
];

export function SiteFooter() {
  return (
    <footer className="bg-zx-bg-alt border-zx-border border-t px-5 pt-14 pb-10 tablet:px-8 laptop:px-12">
      <div className="mx-auto grid max-w-[1280px] grid-cols-2 gap-8 tablet:grid-cols-3 laptop:grid-cols-[2fr_1fr_1fr_1fr_1fr]">
        <div className="col-span-2 tablet:col-span-3 laptop:col-span-1">
          <BrandMark href="/" size="sm" className="mb-[14px]" />
          <p className="text-zx-text-muted m-0 max-w-[260px] text-[13.5px] leading-[1.6]">
            {brand.footerTagline}
          </p>
        </div>

        {COLUMNS.map((column) => (
          <div key={column.title} className="flex flex-col gap-[10px]">
            <span className="text-zx-text-muted mb-1 text-[12px] font-extrabold tracking-[0.08em]">
              {column.title}
            </span>
            {column.items.map((item) => (
              <span
                key={item}
                className="text-zx-text-secondary text-[13.5px]"
              >
                {item}
              </span>
            ))}
          </div>
        ))}
      </div>

      <div className="border-zx-border mx-auto mt-10 flex max-w-[1280px] flex-col gap-3 border-t pt-6 text-[12.5px] tablet:flex-row tablet:items-center tablet:justify-between">
        <span className="text-zx-text-muted">
          © 2026 {brand.name}. All rights reserved.
        </span>
        <DemoFooterNote />
        <span className="text-zx-text-muted flex gap-[18px]">
          {["X", "Instagram", "YouTube", "Discord"].map((social) => (
            <span key={social}>{social}</span>
          ))}
        </span>
      </div>

      <div className="mt-6 text-center">
        <Link href="/app" className="text-[13px] font-bold">
          Open the {brand.name} demo →
        </Link>
      </div>
    </footer>
  );
}
