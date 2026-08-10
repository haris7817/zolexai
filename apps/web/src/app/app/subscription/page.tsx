import type { Metadata } from "next";
import { mockPlan, mockUsage, mockBilling, mockInvoices } from "@/mocks/plan";
import { AppPage, PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";

export const metadata: Metadata = { title: "Subscription" };

/**
 * Subscription — PREUI-09.
 *
 * ⚠️ MOCK. There is NO payment integration in this phase: the billing provider
 * is still an open decision (register D-05), the plan model arrives at M3.10,
 * checkout at M3.13 and webhooks at M3.14. Every button here is inert.
 *
 * The usage block deliberately shows concurrency rather than a generation cap:
 * the public offer is "unlimited", but architecture doc §22 requires internal
 * fair-use protection, and this is the shape those controls take at M3.16.
 */
export default function SubscriptionPage() {
  return (
    <AppPage>
      <PageHeader
        title="Subscription"
        description="Your plan, usage and billing details."
      />

      <div className="grid grid-cols-1 gap-5 laptop:grid-cols-[1fr_320px]">
        <div className="flex flex-col gap-5">
          {/* ── Current plan ─────────────────────────────────────────── */}
          <section className="relative">
            <div
              aria-hidden="true"
              className="absolute -inset-px rounded-[17px] bg-[linear-gradient(135deg,#C6F224,rgba(190,242,8,0.2)_50%,#8CB80A)]"
            />
            <div className="bg-zx-surface relative rounded-[16px] p-6">
              <div className="mb-1 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <span className="text-zx-text text-[18px] font-extrabold">
                    {mockPlan.name}
                  </span>
                  <span className="border-zx-success/40 bg-zx-success/12 text-zx-success rounded-full border px-[10px] py-[3px] text-[11px] font-bold">
                    {mockBilling.status}
                  </span>
                </div>
                <span className="border-zx-border-active bg-zx-primary/20 text-zx-primary-light rounded-full border px-[14px] py-[5px] text-[12px] font-bold">
                  {mockPlan.badge}
                </span>
              </div>

              <div className="mb-1 flex items-baseline gap-[6px]">
                <span className="text-zx-text text-[40px] font-extrabold tracking-[-0.03em]">
                  {mockPlan.price}
                </span>
                <span className="text-zx-text-muted text-[15px] font-semibold">
                  / {mockPlan.interval}
                </span>
              </div>
              <p className="text-zx-text-secondary mt-0 mb-6 text-[13px]">
                Renews on {mockBilling.renewsOn}
              </p>

              <ul className="m-0 mb-6 grid list-none grid-cols-1 gap-[11px] p-0 tablet:grid-cols-2">
                {mockPlan.features.map((feature) => (
                  <li
                    key={feature}
                    className="text-zx-text-secondary flex items-center gap-[10px] text-[13.5px] font-medium"
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

              <div className="flex flex-wrap gap-2">
                <Button variant="ghost" size="md">
                  Manage payment method
                </Button>
                <Button variant="subtle" size="md" className="text-zx-text-muted">
                  Cancel subscription
                </Button>
              </div>
            </div>
          </section>

          {/* ── Usage ────────────────────────────────────────────────── */}
          <section className="bg-zx-surface border-zx-border rounded-zx-lg border p-6">
            <h2 className="text-zx-text m-0 mb-1 text-[16px] font-extrabold">
              Usage this period
            </h2>
            <p className="text-zx-text-muted mt-0 mb-5 text-[12.5px]">
              Fair-use limits keep the generation queue fast for everyone.
            </p>

            <div className="flex flex-col gap-5">
              {mockUsage.map((metric) => (
                <div key={metric.label}>
                  <div className="mb-[6px] flex items-baseline justify-between gap-3">
                    <span className="text-zx-text-secondary text-[13px] font-bold">
                      {metric.label}
                    </span>
                    <span className="text-zx-text text-[13px] font-extrabold">
                      {metric.value}
                    </span>
                  </div>
                  {typeof metric.percent === "number" ? (
                    <div
                      aria-hidden="true"
                      className="mb-[6px] h-[5px] overflow-hidden rounded-[3px] bg-white/7"
                    >
                      <div
                        className="h-full rounded-[3px] bg-[image:var(--zx-gradient-primary)]"
                        style={{ width: `${metric.percent}%` }}
                      />
                    </div>
                  ) : null}
                  <p className="text-zx-text-muted m-0 text-[11.5px]">
                    {metric.detail}
                  </p>
                </div>
              ))}
            </div>
          </section>

          {/* ── Invoices ─────────────────────────────────────────────── */}
          <section className="bg-zx-surface border-zx-border rounded-zx-lg border p-6">
            <h2 className="text-zx-text m-0 mb-4 text-[16px] font-extrabold">
              Billing history
            </h2>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[420px] border-collapse text-left">
                <thead>
                  <tr className="border-zx-border border-b">
                    {["Invoice", "Date", "Amount", "Status", ""].map(
                      (heading) => (
                        <th
                          key={heading}
                          scope="col"
                          className="text-zx-text-muted pb-[10px] text-[11px] font-extrabold tracking-[0.08em] uppercase"
                        >
                          {heading}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {mockInvoices.map((invoice) => (
                    <tr
                      key={invoice.id}
                      className="border-zx-border border-b last:border-b-0"
                    >
                      <td className="text-zx-text py-[13px] text-[12.5px] font-bold">
                        {invoice.id}
                      </td>
                      <td className="text-zx-text-secondary py-[13px] text-[12.5px]">
                        {invoice.date}
                      </td>
                      <td className="text-zx-text-secondary py-[13px] text-[12.5px]">
                        {invoice.amount}
                      </td>
                      <td className="py-[13px]">
                        <span className="text-zx-success text-[12px] font-bold">
                          {invoice.status}
                        </span>
                      </td>
                      <td className="py-[13px] text-right">
                        <button
                          type="button"
                          aria-label={`Download invoice ${invoice.id}`}
                          className="text-zx-text-muted hover:text-zx-text cursor-pointer p-1"
                        >
                          <Icon name="download" size={14} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>

        {/* ── Payment details ────────────────────────────────────────── */}
        <aside className="flex flex-col gap-5">
          <section className="bg-zx-surface border-zx-border rounded-zx-lg h-fit border p-5">
            <h2 className="text-zx-text-muted m-0 mb-4 text-[11px] font-extrabold tracking-[0.11em] uppercase">
              Payment method
            </h2>
            <div className="border-zx-border mb-4 flex items-center gap-3 rounded-[10px] border p-3">
              <span className="bg-zx-surface-elevated text-zx-primary-light flex h-9 w-9 items-center justify-center rounded-lg">
                <Icon name="card" size={16} />
              </span>
              <div>
                <div className="text-zx-text text-[13px] font-bold">
                  {mockBilling.method}
                </div>
                <div className="text-zx-text-muted text-[11.5px]">
                  {mockBilling.methodDetail}
                </div>
              </div>
            </div>

            <h2 className="text-zx-text-muted m-0 mb-2 text-[11px] font-extrabold tracking-[0.11em] uppercase">
              Billing email
            </h2>
            <p className="text-zx-text m-0 text-[13px] font-semibold">
              {mockBilling.billingEmail}
            </p>
          </section>

          <section className="border-zx-border-active rounded-zx-lg border bg-[linear-gradient(120deg,rgba(190,242,8,0.14),rgba(21,21,24,0.6))] p-5">
            <h2 className="text-zx-text m-0 mb-2 text-[14px] font-extrabold">
              Need a hand?
            </h2>
            <p className="text-zx-text-secondary m-0 mb-4 text-[12.5px] leading-[1.55]">
              Questions about billing, invoices or your plan — we usually reply
              the same day.
            </p>
            <Button variant="ghost" size="sm" fullWidth>
              <Icon name="mail" size={14} />
              Contact support
            </Button>
          </section>
        </aside>
      </div>
    </AppPage>
  );
}
