"use client";

import { useState } from "react";
import { brand } from "@/config/brand";
import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

/**
 * Single-open FAQ accordion, matching the approved design's behaviour
 * (clicking the open item closes it).
 *
 * Uses real button/region semantics with aria-expanded and aria-controls —
 * the original design file had neither.
 */

const FAQS = [
  {
    q: `What can I create with ${brand.name}?`,
    a: "Videos from text or images, video transformations and extensions, original music, and full music videos — all from one workspace. New tools are added regularly at no extra cost.",
  },
  {
    q: "Do I need any editing experience?",
    a: "No. Every tool is prompt-driven: describe what you want, adjust a few simple settings, and generate. Advanced controls are there when you want them, hidden when you don't.",
  },
  {
    q: "How long does a generation take?",
    a: "Most generations complete in a few minutes. You can keep browsing, queue multiple jobs, and you'll be notified the moment each one finishes.",
  },
  {
    q: "Who owns the content I generate?",
    a: "You do. Everything you create on your paid plan is yours to download, publish and use commercially.",
  },
  {
    q: "Can I cancel anytime?",
    a: "Yes. Your plan is month-to-month with no lock-in — cancel in one click from your subscription settings.",
  },
];

export function FaqAccordion() {
  const [openIndex, setOpenIndex] = useState(0);

  return (
    <div className="flex flex-col gap-3">
      {FAQS.map((faq, index) => {
        const open = openIndex === index;
        return (
          <div
            key={faq.q}
            className={cn(
              "bg-zx-surface overflow-hidden rounded-[14px] border transition-colors duration-150",
              open ? "border-zx-border-active" : "border-zx-border",
            )}
          >
            <h3 className="m-0">
              <button
                type="button"
                onClick={() => setOpenIndex(open ? -1 : index)}
                aria-expanded={open}
                aria-controls={`faq-panel-${index}`}
                id={`faq-trigger-${index}`}
                className="text-zx-text flex w-full cursor-pointer items-center justify-between gap-4 px-[22px] py-5 text-left text-[14.5px] font-bold tablet:text-[15.5px]"
              >
                {faq.q}
                <span
                  className={cn(
                    "text-zx-accent flex shrink-0 transition-transform duration-200",
                    open && "rotate-180",
                  )}
                >
                  <Icon name="chevron" size={18} />
                </span>
              </button>
            </h3>
            {open ? (
              <div
                id={`faq-panel-${index}`}
                role="region"
                aria-labelledby={`faq-trigger-${index}`}
                className="text-zx-text-secondary px-[22px] pb-5 text-[14px] leading-[1.65] tablet:text-[14.5px]"
              >
                {faq.a}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
