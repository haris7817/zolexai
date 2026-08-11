import type { Metadata } from "next";
import { brand } from "@/config/brand";
import { loadWorkflowCatalog } from "@/features/workflows/catalog.server";
import { mockPlan } from "@/mocks/plan";
import { MarketingNav } from "@/components/marketing/MarketingNav";
import { Hero } from "@/components/marketing/Hero";
import { FaqAccordion } from "@/components/marketing/FaqAccordion";
import { SiteFooter } from "@/components/marketing/SiteFooter";
import { WorkflowCard } from "@/components/workflow/WorkflowCard";
import { ButtonLink } from "@/components/ui/Button";
import { Icon, type IconName } from "@/components/ui/Icon";

export const metadata: Metadata = { title: brand.tagline };

/**
 * Landing page — PREUI-03.
 *
 * Retro-fitted from the original design onto the unified token system and made
 * responsive (the original had no mobile treatment: fixed 48px padding, a 72px
 * headline, `repeat(4,1fr)` grids and a 5-column footer). See ADR 0001.
 *
 * NOTE ON SCOPE: the original design showed SEVEN tool cards, including
 * "AI Editing Tools". That tool is not in the frozen scope (milestones §8.1),
 * so it is deliberately absent — the grid renders the workflow definitions and
 * nothing else, so no surface can imply a seventh tool exists.
 *
 * The catalogue is read from the YAML definitions at render time, NOT fetched
 * from the API. This is a public marketing page: an API deploy or outage must
 * not empty its tool grid. The app itself reads the API at runtime.
 *
 * No provider, model or infrastructure name appears anywhere.
 */
export default async function LandingPage() {
  const workflows = await loadWorkflowCatalog();
  return (
    <div className="bg-zx-bg text-zx-text min-h-screen overflow-x-hidden">
      <MarketingNav />
      <Hero />

      {/* ── Tools ────────────────────────────────────────────────────── */}
      <section
        id="tools"
        className="mx-auto max-w-[1280px] px-5 py-16 tablet:px-8 laptop:px-12 laptop:py-20"
      >
        <SectionHeading
          title="One platform, every medium"
          subtitle="A growing suite of creation tools — new workflows ship regularly."
        />
        <div className="grid grid-cols-1 gap-5 tablet:grid-cols-2 laptop:grid-cols-3">
          {workflows.map((workflow) => (
            <WorkflowCard key={workflow.id} workflow={workflow} tone="detailed" />
          ))}
        </div>
      </section>

      {/* ── How it works ─────────────────────────────────────────────── */}
      <section
        id="how"
        className="bg-zx-bg-alt border-zx-border border-y px-5 py-16 tablet:px-8 laptop:px-12 laptop:py-20"
      >
        <div className="mx-auto max-w-[1080px]">
          <h2 className="m-0 mb-12 text-center text-[30px] font-extrabold tracking-[-0.03em] laptop:mb-14 laptop:text-[42px]">
            From idea to output in three steps
          </h2>
          <div className="grid grid-cols-1 gap-6 tablet:grid-cols-3">
            {[
              {
                n: "1",
                title: "Choose a Tool",
                desc: "Pick from a growing library of video, music and editing workflows.",
              },
              {
                n: "2",
                title: "Describe or Upload",
                desc: "Write a prompt, or bring your own images, video and audio.",
              },
              {
                n: "3",
                title: "Generate",
                desc: "Your creation renders in the background — keep working while it does.",
              },
            ].map((step) => (
              <div key={step.n} className="px-6 py-8 text-center">
                <div className="border-zx-border-active text-zx-primary-light mx-auto mb-5 flex h-[52px] w-[52px] items-center justify-center rounded-[16px] border bg-[linear-gradient(135deg,rgba(190,242,8,0.3),rgba(198,242,36,0.15))] text-[20px] font-extrabold">
                  {step.n}
                </div>
                <div className="mb-2 text-[19px] font-extrabold">
                  {step.title}
                </div>
                <p className="text-zx-text-secondary m-0 text-[14.5px] leading-[1.55]">
                  {step.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Showcase ─────────────────────────────────────────────────── */}
      <section className="mx-auto max-w-[1280px] px-5 py-16 tablet:px-8 laptop:px-12 laptop:py-20">
        <div className="mb-9 flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="m-0 text-[26px] font-extrabold tracking-[-0.03em] laptop:text-[34px]">
            Made with {brand.name}
          </h2>
          <span className="text-zx-accent text-[14px] font-bold">
            Videos · Images · Audio
          </span>
        </div>
        <div className="grid auto-rows-[120px] grid-cols-2 gap-4 laptop:auto-rows-[160px] laptop:grid-cols-4">
          {SHOWCASE.map((item) => (
            <div
              key={item.label}
              className={`border-zx-border relative overflow-hidden rounded-[14px] border ${item.span}`}
              style={{ background: item.bg }}
            >
              <span className="text-zx-text absolute bottom-3 left-[14px] rounded-[7px] bg-[rgba(10,10,11,0.65)] px-[10px] py-1 text-[11.5px] font-bold backdrop-blur-[4px]">
                {item.label}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Benefits ─────────────────────────────────────────────────── */}
      <section className="bg-zx-bg-alt border-zx-border border-y px-5 py-16 tablet:px-8 laptop:px-12 laptop:py-20">
        <div className="mx-auto max-w-[1080px]">
          <h2 className="m-0 mb-12 text-center text-[30px] font-extrabold tracking-[-0.03em] laptop:text-[42px]">
            Built for people who ship
          </h2>
          <div className="grid grid-cols-1 gap-5 tablet:grid-cols-2 laptop:grid-cols-3">
            {BENEFITS.map((benefit) => (
              <div
                key={benefit.title}
                className="bg-zx-surface border-zx-border rounded-zx-lg border p-6"
              >
                <span className="border-zx-border-active text-zx-primary-light mb-4 flex h-11 w-11 items-center justify-center rounded-[13px] border bg-[linear-gradient(135deg,rgba(190,242,8,0.3),rgba(198,242,36,0.12))]">
                  <Icon name={benefit.icon} size={19} />
                </span>
                <div className="mb-2 text-[16px] font-extrabold">
                  {benefit.title}
                </div>
                <p className="text-zx-text-secondary m-0 text-[14px] leading-[1.55]">
                  {benefit.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pricing ──────────────────────────────────────────────────── */}
      <section
        id="pricing"
        className="px-5 py-16 tablet:px-8 laptop:px-12 laptop:pt-20 laptop:pb-24"
      >
        <div className="mx-auto max-w-[520px] text-center">
          <h2 className="m-0 mb-3 text-[30px] font-extrabold tracking-[-0.03em] laptop:text-[42px]">
            Simple pricing
          </h2>
          <p className="text-zx-text-secondary m-0 mb-11 text-[16px]">
            One plan. Everything included.
          </p>

          <div className="relative text-left">
            <div
              aria-hidden="true"
              className="absolute -inset-px rounded-[21px] bg-[linear-gradient(135deg,#C6F224,rgba(190,242,8,0.2)_50%,#33430D)]"
            />
            <div className="bg-zx-surface relative rounded-[20px] p-7 laptop:p-9">
              <div className="mb-2 flex items-center justify-between gap-3">
                <span className="text-[18px] font-extrabold">
                  {mockPlan.name}
                </span>
                <span className="border-zx-border-active bg-zx-primary/20 text-zx-primary-light rounded-full border px-[14px] py-[5px] text-[12px] font-bold">
                  {mockPlan.badge}
                </span>
              </div>

              <div className="mb-[26px] flex items-baseline gap-[6px]">
                <span className="text-[44px] font-extrabold tracking-[-0.03em] laptop:text-[52px]">
                  {mockPlan.price}
                </span>
                <span className="text-zx-text-muted text-[15px] font-semibold">
                  / {mockPlan.interval}
                </span>
              </div>

              <ul className="m-0 mb-8 flex list-none flex-col gap-[13px] p-0">
                {mockPlan.features.map((feature) => (
                  <li
                    key={feature}
                    className="text-zx-text-secondary flex items-center gap-[11px] text-[14.5px] font-medium"
                  >
                    <span
                      aria-hidden="true"
                      className="bg-zx-primary/25 text-zx-primary-light flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-full"
                    >
                      <Icon name="check" size={11} />
                    </span>
                    {feature}
                  </li>
                ))}
              </ul>

              <ButtonLink href="/app" variant="primary" size="lg" fullWidth>
                Start Creating
              </ButtonLink>
            </div>
          </div>
        </div>
      </section>

      {/* ── FAQ ──────────────────────────────────────────────────────── */}
      <section className="px-5 pb-20 tablet:px-8 laptop:px-12 laptop:pb-24">
        <div className="mx-auto max-w-[720px]">
          <h2 className="m-0 mb-10 text-center text-[26px] font-extrabold tracking-[-0.03em] laptop:text-[34px]">
            Frequently asked questions
          </h2>
          <FaqAccordion />
        </div>
      </section>

      <SiteFooter />
    </div>
  );
}

function SectionHeading({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) {
  return (
    <div className="mb-12 text-center laptop:mb-14">
      <h2 className="m-0 mb-[14px] text-[30px] font-extrabold tracking-[-0.03em] laptop:text-[42px]">
        {title}
      </h2>
      <p className="text-zx-text-secondary m-0 text-[16px] laptop:text-[17px]">
        {subtitle}
      </p>
    </div>
  );
}

const g = (a: string, b: string) => `linear-gradient(140deg, ${a}, ${b})`;

const SHOWCASE = [
  {
    span: "col-span-2",
    bg: g("#222C10", "#121808"),
    label: "Video · Neon city flythrough",
  },
  { span: "col-span-1", bg: g("#2C3A0B", "#141A0C"), label: "Image · Portrait study" },
  { span: "col-span-1", bg: g("#18200A", "#0E1207"), label: "Audio · Ambient score" },
  { span: "col-span-1", bg: g("#101318", "#242A32"), label: "Video · Product loop" },
  {
    span: "col-span-2",
    bg: g("#182008", "#33430D"),
    label: "Music video · Synthwave",
  },
  { span: "col-span-1", bg: g("#28340D", "#10160A"), label: "Image · Concept art" },
];

const BENEFITS: { icon: IconName; title: string; desc: string }[] = [
  {
    icon: "wand",
    title: "No editing experience needed",
    desc: "Every tool is prompt-driven. Advanced controls are there when you want them, hidden when you don't.",
  },
  {
    icon: "history",
    title: "Work while it renders",
    desc: "Queue several generations and keep creating — nothing locks you on a loading screen.",
  },
  {
    icon: "folder",
    title: "Everything in one library",
    desc: "Uploads and results live together, ready to reuse in any workflow.",
  },
  {
    icon: "extend",
    title: "Go longer, seamlessly",
    desc: "Extend any generated clip again and again without losing the original.",
  },
  {
    icon: "grid",
    title: "New tools as they ship",
    desc: "Your plan includes every workflow we add — no upgrade, no extra cost.",
  },
  {
    icon: "download",
    title: "Yours to publish",
    desc: "Download what you create and use it commercially on your paid plan.",
  },
];
