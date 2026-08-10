"use client";

import { useState } from "react";
import Link from "next/link";
import { mockUser } from "@/mocks/user";
import { WORKFLOW_LIST } from "@/features/workflows/registry";
import { AppPage, PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";
import { Icon, type IconName } from "@/components/ui/Icon";
import {
  SectionLabel,
  TextField,
  ToggleField,
  OptionChip,
} from "@/components/ui/Controls";
import { cn } from "@/lib/cn";

/**
 * Settings — PREUI-10.
 *
 * ⚠️ MOCK. Nothing saves. There is no account system in this phase — profile
 * and security arrive at M3.18 and M3.02, appearance preferences are not yet
 * scoped. Every field is here so the client can approve the structure and
 * grouping before any of it is built.
 */

type SectionId =
  | "profile"
  | "account"
  | "preferences"
  | "generation"
  | "appearance";

const SECTIONS: { id: SectionId; label: string; icon: IconName }[] = [
  { id: "profile", label: "Profile", icon: "user" },
  { id: "account", label: "Account & security", icon: "shield" },
  { id: "preferences", label: "Preferences", icon: "bell" },
  { id: "generation", label: "Generation defaults", icon: "sliders" },
  { id: "appearance", label: "Appearance", icon: "palette" },
];

export function SettingsView() {
  const [active, setActive] = useState<SectionId>("profile");

  return (
    <AppPage>
      <PageHeader
        title="Settings"
        description="Manage your profile, preferences and generation defaults."
      />

      <div className="grid grid-cols-1 gap-6 laptop:grid-cols-[220px_1fr]">
        {/* ── Sub navigation ─────────────────────────────────────────── */}
        <nav
          aria-label="Settings sections"
          className="flex gap-1 overflow-x-auto laptop:flex-col laptop:overflow-visible"
        >
          {SECTIONS.map((section) => {
            const selected = section.id === active;
            return (
              <button
                key={section.id}
                type="button"
                aria-current={selected ? "page" : undefined}
                onClick={() => setActive(section.id)}
                className={cn(
                  "rounded-zx-sm flex shrink-0 cursor-pointer items-center gap-[10px] px-3 py-[10px] text-[13px] whitespace-nowrap transition-colors duration-150",
                  selected
                    ? "bg-zx-primary/10 text-zx-text font-extrabold shadow-[inset_2px_0_0_var(--zx-accent)]"
                    : "text-zx-text-secondary hover:bg-zx-surface-hover hover:text-zx-text font-semibold",
                )}
              >
                <Icon name={section.icon} size={15} />
                {section.label}
              </button>
            );
          })}

          <Link
            href="/app/subscription"
            className="rounded-zx-sm text-zx-text-secondary hover:bg-zx-surface-hover hover:text-zx-text flex shrink-0 items-center gap-[10px] px-3 py-[10px] text-[13px] font-semibold whitespace-nowrap"
          >
            <Icon name="card" size={15} />
            Billing
            <Icon name="arrowUpRight" size={13} />
          </Link>
        </nav>

        <div className="min-w-0">
          {active === "profile" ? <ProfileSection /> : null}
          {active === "account" ? <AccountSection /> : null}
          {active === "preferences" ? <PreferencesSection /> : null}
          {active === "generation" ? <GenerationSection /> : null}
          {active === "appearance" ? <AppearanceSection /> : null}
        </div>
      </div>
    </AppPage>
  );
}

function Panel({
  title,
  description,
  children,
  footer = true,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
  footer?: boolean;
}) {
  return (
    <section className="bg-zx-surface border-zx-border rounded-zx-lg animate-zx-fade-up mb-5 border p-6">
      <h2 className="text-zx-text m-0 mb-1 text-[16px] font-extrabold">
        {title}
      </h2>
      <p className="text-zx-text-muted mt-0 mb-5 text-[12.5px]">{description}</p>
      {children}
      {footer ? (
        <div className="border-zx-border mt-6 flex justify-end border-t pt-5">
          <Button variant="primary" size="md">
            Save changes
          </Button>
        </div>
      ) : null}
    </section>
  );
}

function Field({
  id,
  label,
  defaultValue,
  type = "text",
  hint,
}: {
  id: string;
  label: string;
  defaultValue?: string;
  type?: string;
  hint?: string;
}) {
  return (
    <div className="mb-4">
      <SectionLabel as="label" htmlFor={id}>
        {label}
      </SectionLabel>
      <TextField id={id} type={type} defaultValue={defaultValue} />
      {hint ? (
        <p className="text-zx-text-muted mt-[6px] mb-0 text-[11.5px]">{hint}</p>
      ) : null}
    </div>
  );
}

function ProfileSection() {
  return (
    <Panel
      title="Profile"
      description="How you appear across ZolexAI."
    >
      <div className="mb-6 flex items-center gap-4">
        <span
          aria-hidden="true"
          className="flex h-16 w-16 items-center justify-center rounded-full bg-[linear-gradient(135deg,var(--zx-accent),#6E8F05)] text-[20px] font-extrabold text-zx-on-primary"
        >
          {mockUser.initials}
        </span>
        <div>
          <Button variant="ghost" size="sm">
            <Icon name="upload" size={13} />
            Change photo
          </Button>
          <p className="text-zx-text-muted mt-2 mb-0 text-[11.5px]">
            JPG or PNG, up to 2 MB.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-x-4 tablet:grid-cols-2">
        <Field id="set-name" label="Display name" defaultValue={mockUser.name} />
        <Field
          id="set-email"
          label="Email"
          type="email"
          defaultValue={mockUser.email}
        />
        <Field id="set-location" label="Location" defaultValue={mockUser.location} />
        <Field id="set-timezone" label="Time zone" defaultValue={mockUser.timezone} />
      </div>
    </Panel>
  );
}

function AccountSection() {
  return (
    <>
      <Panel
        title="Password"
        description="Choose a strong password you don't use anywhere else."
      >
        <Field id="set-current" label="Current password" type="password" />
        <div className="grid grid-cols-1 gap-x-4 tablet:grid-cols-2">
          <Field id="set-new" label="New password" type="password" />
          <Field id="set-confirm" label="Confirm new password" type="password" />
        </div>
      </Panel>

      <Panel
        title="Sessions"
        description="Devices currently signed in to your account."
        footer={false}
      >
        <div className="border-zx-border flex items-center justify-between gap-4 rounded-[10px] border p-4">
          <div>
            <div className="text-zx-text text-[13px] font-bold">
              Chrome on Windows
            </div>
            <div className="text-zx-text-muted mt-[2px] text-[11.5px]">
              Berlin, Germany · Active now
            </div>
          </div>
          <span className="text-zx-success text-[11.5px] font-bold">
            This device
          </span>
        </div>
      </Panel>

      <Panel
        title="Danger zone"
        description="Permanently delete your account and all generated media."
        footer={false}
      >
        <Button
          variant="ghost"
          size="md"
          className="border-zx-error/40 text-zx-error hover:text-zx-error"
        >
          <Icon name="trash" size={14} />
          Delete account
        </Button>
      </Panel>
    </>
  );
}

function PreferencesSection() {
  const [emailOnComplete, setEmailOnComplete] = useState(true);
  const [emailOnFail, setEmailOnFail] = useState(true);
  const [productNews, setProductNews] = useState(false);
  const [browserAlerts, setBrowserAlerts] = useState(true);

  return (
    <Panel
      title="Notifications"
      description="Choose when ZolexAI should get in touch."
    >
      <div className="flex flex-col gap-4">
        <ToggleField
          id="pref-complete"
          label="Email me when a generation completes"
          checked={emailOnComplete}
          onChange={setEmailOnComplete}
        />
        <ToggleField
          id="pref-fail"
          label="Email me when a generation fails"
          checked={emailOnFail}
          onChange={setEmailOnFail}
        />
        <ToggleField
          id="pref-browser"
          label="Show browser notifications"
          checked={browserAlerts}
          onChange={setBrowserAlerts}
        />
        <ToggleField
          id="pref-news"
          label="Product news and new tool announcements"
          checked={productNews}
          onChange={setProductNews}
        />
      </div>
    </Panel>
  );
}

function GenerationSection() {
  const [workflowId, setWorkflowId] = useState(WORKFLOW_LIST[0].id);
  const [autoOpen, setAutoOpen] = useState(true);
  const workflow =
    WORKFLOW_LIST.find((item) => item.id === workflowId) ?? WORKFLOW_LIST[0];

  return (
    <Panel
      title="Generation defaults"
      description="Settings applied when you open a new workflow."
    >
      <SectionLabel>Default tool</SectionLabel>
      <div className="mb-6 flex flex-wrap gap-2">
        {WORKFLOW_LIST.map((item) => (
          <OptionChip
            key={item.id}
            selected={item.id === workflowId}
            onClick={() => setWorkflowId(item.id)}
            className="inline-flex items-center gap-[6px] px-[14px] py-[9px] text-[12px]"
          >
            <Icon name={item.icon} size={13} />
            {item.name}
          </OptionChip>
        ))}
      </div>

      {/* Options come from the registry, so this section is correct for whichever
          tool is selected — the same mechanism the workspace panel uses. */}
      <SectionLabel>Default duration</SectionLabel>
      <div className="mb-6 flex flex-wrap gap-2">
        {workflow.supportedDurations.map((duration, index) => (
          <OptionChip
            key={duration}
            selected={index === 0}
            onClick={noop}
            className="px-4 py-[9px] text-[12px]"
          >
            {duration}
          </OptionChip>
        ))}
      </div>

      {workflow.supportedAspectRatios.length > 0 ? (
        <>
          <SectionLabel>Default aspect ratio</SectionLabel>
          <div className="mb-6 flex flex-wrap gap-2">
            {workflow.supportedAspectRatios.map((ratio, index) => (
              <OptionChip
                key={ratio}
                selected={index === 0}
                onClick={noop}
                className="px-4 py-[9px] text-[12px]"
              >
                {ratio}
              </OptionChip>
            ))}
          </div>
        </>
      ) : null}

      <ToggleField
        id="pref-autoopen"
        label="Open the result automatically when a generation completes"
        checked={autoOpen}
        onChange={setAutoOpen}
      />
    </Panel>
  );
}

function AppearanceSection() {
  const [reduceMotion, setReduceMotion] = useState(false);
  const [compactStrip, setCompactStrip] = useState(false);

  return (
    <Panel
      title="Appearance"
      description="ZolexAI is designed dark-first. Light mode is not part of the current scope."
    >
      <SectionLabel>Theme</SectionLabel>
      <div className="mb-6 flex flex-wrap gap-2">
        <OptionChip selected onClick={noop} className="px-4 py-[9px] text-[12px]">
          Dark
        </OptionChip>
        <OptionChip
          selected={false}
          onClick={noop}
          disabled
          className="px-4 py-[9px] text-[12px] opacity-45"
        >
          Light — coming later
        </OptionChip>
      </div>

      <div className="flex flex-col gap-4">
        <ToggleField
          id="pref-motion"
          label="Reduce motion and background animation"
          checked={reduceMotion}
          onChange={setReduceMotion}
        />
        <ToggleField
          id="pref-strip"
          label="Compact generation strip"
          checked={compactStrip}
          onChange={setCompactStrip}
        />
      </div>
    </Panel>
  );
}

function noop() {
  /* mock control — nothing persists in this phase */
}
