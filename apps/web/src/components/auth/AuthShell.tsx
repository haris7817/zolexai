import Link from "next/link";
import type { ReactNode } from "react";
import { brand } from "@/config/brand";
import { BrandMark } from "@/components/navigation/BrandMark";
import { ButtonLink } from "@/components/ui/Button";
import { SectionLabel, TextField } from "@/components/ui/Controls";
import { Icon } from "@/components/ui/Icon";

/**
 * Auth visual direction — PREUI-11.
 *
 * ⚠️ MOCK. There is NO authentication in this phase. Nothing is validated,
 * nothing is stored, and the submit button simply routes to /app so the demo
 * never dead-ends. Real accounts arrive at M3.02 (registration/login),
 * M3.03 (password reset) and M3.04 (ownership).
 *
 * Included so the client can approve the entry experience alongside the rest
 * of the product, not to imply a working account system.
 */
export function AuthShell({
  title,
  subtitle,
  submitLabel,
  fields,
  footer,
}: {
  title: string;
  subtitle: string;
  submitLabel: string;
  fields: ReactNode;
  footer: ReactNode;
}) {
  return (
    <div className="bg-zx-bg text-zx-text relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-5 py-12">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute top-[-220px] left-1/2 h-[560px] w-[860px] max-w-[150vw] -translate-x-1/2 bg-[radial-gradient(ellipse_at_center,rgba(190,242,8,0.14)_0%,rgba(190,242,8,0)_65%)]"
      />

      <div className="relative w-full max-w-[400px]">
        <div className="mb-8 flex justify-center">
          <BrandMark href="/" size="lg" />
        </div>

        <div className="relative">
          <div
            aria-hidden="true"
            className="absolute -inset-px rounded-[19px] bg-[linear-gradient(135deg,rgba(198,242,36,0.45),rgba(190,242,8,0.12)_50%,rgba(198,242,36,0.3))]"
          />
          <div className="bg-zx-surface relative rounded-[18px] p-7">
            <h1 className="text-zx-text m-0 mb-[6px] text-[22px] font-extrabold tracking-[-0.02em]">
              {title}
            </h1>
            <p className="text-zx-text-secondary mt-0 mb-6 text-[13.5px]">
              {subtitle}
            </p>

            {fields}

            <ButtonLink
              href="/app"
              variant="primary"
              size="lg"
              fullWidth
              className="mt-2"
            >
              {submitLabel}
            </ButtonLink>

            <div className="my-5 flex items-center gap-3">
              <span className="bg-zx-border h-px flex-1" />
              <span className="text-zx-text-muted text-[11.5px] font-bold">
                OR
              </span>
              <span className="bg-zx-border h-px flex-1" />
            </div>

            <ButtonLink href="/app" variant="ghost" size="lg" fullWidth>
              <Icon name="mail" size={15} />
              Continue with email link
            </ButtonLink>
          </div>
        </div>

        <p className="text-zx-text-secondary mt-6 text-center text-[13px]">
          {footer}
        </p>

        <p className="text-zx-text-muted mt-6 text-center text-[11.5px] leading-[1.5]">
          Interactive product preview — accounts are not connected in this demo.
        </p>

        <div className="mt-4 text-center">
          <Link href="/" className="text-[12.5px] font-bold">
            ← Back to {brand.name}
          </Link>
        </div>
      </div>
    </div>
  );
}

export function AuthField({
  id,
  label,
  type = "text",
  placeholder,
}: {
  id: string;
  label: string;
  type?: string;
  placeholder?: string;
}) {
  return (
    <div className="mb-4">
      <SectionLabel as="label" htmlFor={id}>
        {label}
      </SectionLabel>
      <TextField id={id} type={type} placeholder={placeholder} />
    </div>
  );
}
