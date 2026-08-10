import { brand } from "@/config/brand";
import { ButtonLink } from "@/components/ui/Button";

/**
 * Landing hero + product mockup.
 *
 * Responsive work the original design did not have: the 72px headline is now a
 * clamp, and the three-pane mockup drops its side rails below laptop so the
 * canvas stays legible rather than compressing to a sliver.
 */
export function Hero() {
  return (
    <section className="relative px-5 pt-12 pb-16 text-center tablet:px-8 laptop:px-12 laptop:pt-[88px] laptop:pb-24">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute top-[-180px] left-1/2 h-[600px] w-[900px] max-w-[140vw] -translate-x-1/2 bg-[radial-gradient(ellipse_at_center,rgba(109,61,245,0.28)_0%,rgba(109,61,245,0)_65%)]"
      />

      <div className="relative mx-auto max-w-[820px]">
        <div className="border-zx-border-active bg-zx-primary/10 text-zx-primary-light mb-7 inline-flex items-center gap-2 rounded-full border px-4 py-[7px] text-[12px] font-semibold tablet:text-[13px]">
          <span
            aria-hidden="true"
            className="bg-zx-accent inline-block h-[6px] w-[6px] rounded-full"
          />
          One workspace. Every AI creation tool.
        </div>

        <h1 className="m-0 mb-6 text-[clamp(38px,8vw,72px)] leading-[1.04] font-extrabold tracking-[-0.035em]">
          Turn Ideas Into
          <br />
          <span className="bg-[linear-gradient(120deg,#9D6BFF,#C4B5FD_60%,#7E52FF)] bg-clip-text text-transparent">
            Motion.
          </span>
        </h1>

        <p className="text-zx-text-secondary mx-auto mb-10 max-w-[560px] text-[16px] leading-[1.6] text-pretty tablet:text-[19px]">
          {brand.description} No timeline anxiety, no technical setup — just
          describe and generate.
        </p>

        <div className="flex flex-col justify-center gap-[14px] tablet:flex-row">
          <ButtonLink href="/app" variant="primary" size="lg">
            Start Creating
          </ButtonLink>
          <ButtonLink
            href="#tools"
            variant="ghost"
            size="lg"
            className="bg-white/5 text-zx-text"
          >
            Explore Tools
          </ButtonLink>
        </div>
      </div>

      <WorkspaceMockup />
    </section>
  );
}

function WorkspaceMockup() {
  return (
    <div className="relative mx-auto mt-14 max-w-[1080px] laptop:mt-[72px]">
      <div
        aria-hidden="true"
        className="absolute -inset-[2px] rounded-[22px] bg-[linear-gradient(135deg,rgba(157,107,255,0.5),rgba(109,61,245,0.1)_40%,rgba(157,107,255,0.35))] blur-[1px]"
      />
      <div className="bg-zx-surface border-zx-border relative overflow-hidden rounded-[20px] border shadow-[0_40px_120px_rgba(0,0,0,0.6)]">
        <div className="border-zx-border flex items-center gap-[6px] border-b px-[18px] py-3">
          {[0, 1, 2].map((dot) => (
            <span
              key={dot}
              aria-hidden="true"
              className="h-[10px] w-[10px] rounded-full bg-[#2A2545]"
            />
          ))}
        </div>

        <div className="grid h-[320px] grid-cols-1 laptop:h-[480px] laptop:grid-cols-[200px_1fr_260px]">
          {/* Side rails are decorative — hidden below laptop so the canvas keeps its size */}
          <div className="border-zx-border hidden flex-col gap-[6px] border-r p-[18px] px-[14px] text-left laptop:flex">
            <div className="mb-[14px] flex items-center gap-2">
              <span
                aria-hidden="true"
                className="h-[22px] w-[22px] rounded-md bg-[image:var(--zx-gradient-primary)]"
              />
              <span className="text-[13px] font-extrabold">{brand.name}</span>
            </div>
            <div className="bg-zx-primary/22 text-zx-primary-light rounded-lg px-[10px] py-2 text-[12px] font-bold">
              Text to Video
            </div>
            {["Image to Video", "Music", "Music Video", "All Tools"].map(
              (label) => (
                <div
                  key={label}
                  className="text-zx-text-muted px-[10px] py-2 text-[12px] font-semibold"
                >
                  {label}
                </div>
              ),
            )}
          </div>

          <div className="flex items-center justify-center bg-[radial-gradient(ellipse_at_50%_30%,rgba(109,61,245,0.12),transparent_70%)] p-5 laptop:p-7">
            <div className="border-zx-border relative flex aspect-video w-full max-w-[560px] items-center justify-center overflow-hidden rounded-[14px] border bg-[linear-gradient(140deg,#1C1838_0%,#2A1F55_50%,#171331_100%)]">
              <div
                aria-hidden="true"
                className="absolute inset-0 bg-[radial-gradient(circle_at_70%_20%,rgba(157,107,255,0.35),transparent_55%)]"
              />
              <div className="relative flex h-[58px] w-[58px] items-center justify-center rounded-full border border-white/25 bg-white/10 backdrop-blur-[6px]">
                <span
                  aria-hidden="true"
                  className="ml-[4px] h-0 w-0 border-y-[10px] border-l-[16px] border-y-transparent border-l-white"
                />
              </div>
            </div>
          </div>

          <div className="border-zx-border hidden flex-col gap-3 border-l p-[18px] px-4 text-left laptop:flex">
            <div className="text-zx-text-muted text-[11px] font-bold tracking-[0.08em]">
              PROMPT
            </div>
            <div className="border-zx-border text-zx-text-secondary rounded-[10px] border bg-[#171331] p-[10px] text-[11.5px] leading-[1.5]">
              A slow cinematic dolly shot through a neon-lit city at dusk, rain
              reflections…
            </div>
            <div className="text-zx-text-muted text-[11px] font-bold tracking-[0.08em]">
              DURATION
            </div>
            <div className="flex gap-[6px]">
              <span className="border-zx-border-active bg-zx-primary/25 text-zx-primary-light rounded-[7px] border px-[10px] py-[5px] text-[11px] font-bold">
                5s
              </span>
              {["10s", "15s"].map((value) => (
                <span
                  key={value}
                  className="border-zx-border text-zx-text-muted rounded-[7px] border px-[10px] py-[5px] text-[11px] font-bold"
                >
                  {value}
                </span>
              ))}
            </div>
            <div className="mt-auto rounded-[10px] bg-[image:var(--zx-gradient-primary)] p-[11px] text-center text-[13px] font-extrabold text-white">
              Generate
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
