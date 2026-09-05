import Image from "next/image";
import { brand } from "@/config/brand";
import { HERO_PREVIEW_VIDEO } from "@/config/marketing-videos";
import { ButtonLink } from "@/components/ui/Button";
import { HeroPreviewVideo } from "@/components/marketing/PreviewVideo";

/**
 * Landing hero + product mockup.
 *
 * The first fold follows the client's latest mobile reference (CR-002): the
 * real ZolexAI lockup sits prominently at the top on the black ground, then
 * the "CREATE THE IMPOSSIBLE." headline, short supporting copy and a strong
 * Start Creating CTA. On mobile the badge pill steps aside so the fold is
 * exactly logo → headline → copy → CTA; from tablet up it returns between the
 * logo and the headline.
 *
 * Responsive work the original design did not have: the 72px headline is now a
 * clamp, and the three-pane mockup drops its side rails below laptop so the
 * canvas stays legible rather than compressing to a sliver.
 */
export function Hero() {
  return (
    <section className="relative px-5 pt-10 pb-16 text-center tablet:px-8 laptop:px-12 laptop:pt-14 laptop:pb-24">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute top-[-180px] left-1/2 h-[600px] w-[900px] max-w-[140vw] -translate-x-1/2 bg-[radial-gradient(ellipse_at_center,rgba(190,242,8,0.16)_0%,rgba(190,242,8,0)_65%)]"
      />

      <div className="relative mx-auto max-w-[820px]">
        {/* The client-provided lockup (monogram, wordmark, tagline) on
            transparency. `priority` because this is the page's LCP element. */}
        <Image
          src="/brand/logo-full.png"
          alt={`${brand.name} — Create Without Limits`}
          width={900}
          height={838}
          priority
          unoptimized
          className="mx-auto mb-6 h-[clamp(150px,34vw,216px)] w-auto drop-shadow-[0_0_44px_rgba(190,242,8,0.28)] tablet:mb-7"
        />

        <div className="border-zx-border-active bg-zx-primary/10 text-zx-primary-light mb-7 hidden items-center gap-2 rounded-full border px-4 py-[7px] text-[12px] font-semibold tablet:inline-flex tablet:text-[13px]">
          <span
            aria-hidden="true"
            className="bg-zx-accent inline-block h-[6px] w-[6px] rounded-full"
          />
          One workspace. Every AI creation tool.
        </div>

        {/* Uppercase display type is wider per character than the previous
            mixed-case headline, so the clamp floor is raised and the ceiling
            eased to keep "IMPOSSIBLE." on one line down to 390px. */}
        <h1 className="m-0 mb-6 text-[clamp(36px,7.5vw,72px)] leading-[1.02] font-extrabold tracking-[-0.03em] uppercase">
          Create The
          <br />
          <span className="bg-[linear-gradient(120deg,#D2FF3A,#E6FF9C_55%,#A9DE00)] bg-clip-text text-transparent">
            Impossible.
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
        className="absolute -inset-[2px] rounded-[22px] bg-[linear-gradient(135deg,rgba(198,242,36,0.5),rgba(190,242,8,0.1)_40%,rgba(198,242,36,0.35))] blur-[1px]"
      />
      <div className="bg-zx-surface border-zx-border relative overflow-hidden rounded-[20px] border shadow-[0_40px_120px_rgba(0,0,0,0.6)]">
        <div className="border-zx-border flex items-center gap-[6px] border-b px-[18px] py-3">
          {[0, 1, 2].map((dot) => (
            <span
              key={dot}
              aria-hidden="true"
              className="h-[10px] w-[10px] rounded-full bg-[#26291F]"
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

          <div className="flex items-center justify-center bg-[radial-gradient(ellipse_at_50%_30%,rgba(190,242,8,0.12),transparent_70%)] p-5 laptop:p-7">
            <div className="border-zx-border relative flex aspect-video w-full max-w-[560px] items-center justify-center overflow-hidden rounded-[14px] border bg-[linear-gradient(140deg,#141A08_0%,#1E2A08_50%,#10160A_100%)]">
              <div
                aria-hidden="true"
                className="absolute inset-0 bg-[radial-gradient(circle_at_70%_20%,rgba(198,242,36,0.16),transparent_60%)]"
              />
              <HeroPreviewVideo src={HERO_PREVIEW_VIDEO} label="ZolexAI creation showcase" />
            </div>
          </div>

          <div className="border-zx-border hidden flex-col gap-3 border-l p-[18px] px-4 text-left laptop:flex">
            <div className="text-zx-text-muted text-[11px] font-bold tracking-[0.08em]">
              PROMPT
            </div>
            <div className="border-zx-border text-zx-text-secondary rounded-[10px] border bg-[#10160A] p-[10px] text-[11.5px] leading-[1.5]">
              A slow cinematic dolly shot through a neon-lit city at dusk, rain
              reflections…
            </div>
            <div className="text-zx-text-muted text-[11px] font-bold tracking-[0.08em]">
              DURATION
            </div>
            <div className="flex flex-wrap gap-[6px]">
              <span className="border-zx-border-active bg-zx-primary/25 text-zx-primary-light rounded-[7px] border px-[10px] py-[5px] text-[11px] font-bold">
                5s
              </span>
              {["10s", "15s", "30s", "60s"].map((value) => (
                <span
                  key={value}
                  className="border-zx-border text-zx-text-muted rounded-[7px] border px-[10px] py-[5px] text-[11px] font-bold"
                >
                  {value}
                </span>
              ))}
            </div>
            <div className="mt-auto rounded-[10px] bg-[image:var(--zx-gradient-primary)] p-[11px] text-center text-[13px] font-extrabold text-zx-on-primary">
              Generate
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
