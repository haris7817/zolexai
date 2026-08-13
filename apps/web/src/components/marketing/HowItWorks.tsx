import { Icon, type IconName } from "@/components/ui/Icon";
import { Reveal } from "./Reveal";

/**
 * The product flow, as a connected sequence rather than a row of cards.
 *
 * Numbering is used here because the content genuinely IS ordered — you
 * cannot preview before you generate — which is the only thing that justifies
 * step markers. The connector carries the same meaning: it runs between
 * consecutive steps and stops at the last one, so the line says "these follow
 * each other" rather than decorating.
 *
 * Two layouts, one markup: a horizontal rail from laptop up, where the
 * connector is a horizontal line between medallions; and a stacked list
 * below, where it becomes a vertical line down the left gutter. Both come
 * from the same list, so a step cannot exist in one layout and not the other.
 *
 * Steps stagger in on scroll (80ms apart) so the eye is walked along the
 * sequence in order — the one place on this page where animation carries
 * meaning rather than polish.
 */

const STEPS: { title: string; desc: string; icon: IconName }[] = [
  {
    icon: "grid",
    title: "Pick your tool",
    desc: "Video, music or a music video — each one is a single screen with no setup.",
  },
  {
    icon: "wand",
    title: "Describe or upload",
    desc: "Write what you want, or bring an image, a clip or a track to build from.",
  },
  {
    icon: "sparkles",
    title: "We generate it",
    desc: "Your creation renders in the background while you keep working.",
  },
  {
    icon: "play",
    title: "Preview instantly",
    desc: "Watch the result the moment it lands, right where you made it.",
  },
  {
    icon: "extend",
    title: "Download or go longer",
    desc: "Publish it, extend it, or spin a variation from the same settings.",
  },
];

export function HowItWorks() {
  return (
    <section
      id="how"
      className="bg-zx-bg-alt border-zx-border border-y px-5 py-16 tablet:px-8 laptop:px-12 laptop:py-20"
    >
      <div className="mx-auto max-w-[1180px]">
        <Reveal>
          {/* CR-001 — the client's exact copy. */}
          <h2 className="m-0 mb-3 text-center text-[30px] font-extrabold tracking-[-0.03em] laptop:text-[42px]">
            IMAGINE IT. GENERATE IT. GO VIRAL.
          </h2>
          <p className="text-zx-text-secondary mx-auto mb-12 max-w-[560px] text-center text-[16px] laptop:mb-16 laptop:text-[17px]">
            Five steps from an idea to something you can post.
          </p>
        </Reveal>

        <ol className="relative m-0 grid list-none grid-cols-1 gap-0 p-0 laptop:grid-cols-5 laptop:gap-5">
          {/* Horizontal rail, laptop and up. Sits behind the medallions and
              spans between the first and last one rather than the full width,
              so the line never dangles past the sequence it connects. */}
          <span
            aria-hidden="true"
            className="pointer-events-none absolute top-[30px] right-[10%] left-[10%] hidden h-px bg-[linear-gradient(90deg,transparent,var(--zx-border-active)_12%,var(--zx-border-active)_88%,transparent)] laptop:block"
          />

          {STEPS.map((step, index) => (
            <Reveal
              as="li"
              key={step.title}
              delay={index * 80}
              className="relative pb-9 pl-[70px] last:pb-0 laptop:p-0 laptop:text-center"
            >
              {/* Vertical connector for the stacked layout — drawn from each
                  step down to the next, so the last one ends the line. */}
              <span
                aria-hidden="true"
                className="bg-zx-border absolute top-[52px] bottom-0 left-[25px] w-px laptop:hidden"
              />

              <span
                aria-hidden="true"
                className="border-zx-border-active bg-zx-bg-alt text-zx-primary-light absolute top-0 left-0 z-[1] flex h-[52px] w-[52px] items-center justify-center rounded-[17px] border bg-[linear-gradient(135deg,rgba(190,242,8,0.28),rgba(198,242,36,0.1))] laptop:relative laptop:mx-auto laptop:mb-5"
              >
                <Icon name={step.icon} size={21} />
                <span className="bg-zx-primary text-zx-on-primary absolute -top-[6px] -right-[6px] flex h-[22px] w-[22px] items-center justify-center rounded-full text-[11px] font-extrabold">
                  {index + 1}
                </span>
              </span>

              <h3 className="text-zx-text m-0 mb-[6px] pt-[13px] text-[17px] font-extrabold laptop:pt-0 laptop:text-[16.5px]">
                {step.title}
              </h3>
              <p className="text-zx-text-secondary m-0 text-[14px] leading-[1.55] laptop:text-[13.5px]">
                {step.desc}
              </p>
            </Reveal>
          ))}
        </ol>
      </div>
    </section>
  );
}
