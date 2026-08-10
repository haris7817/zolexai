/**
 * MOCK — PRE-M1 demo only.
 *
 * Presents the planned commercial offer for design approval. There is NO
 * payment integration in this phase — the billing provider is still an open
 * decision (register D-05), and checkout arrives at M3.13.
 */

export const mockPlan = {
  name: "ZolexAI Unlimited",
  price: "$70",
  interval: "month",
  badge: "All access",
  description: "One plan. Every creation tool. New workflows as they ship.",
  features: [
    "Access to all creation tools",
    "AI video generation",
    "Music creation",
    "Video extensions",
    "Generation history",
    "Media library",
    "New tools as released",
  ],
} as const;

/**
 * Usage meters. The public offer is "unlimited", but architecture doc §22
 * requires internal fair-use protection, so the UI is designed to show
 * concurrency rather than a hard cap — the shape those controls will take
 * at M3.16.
 */
export const mockUsage = [
  {
    label: "Generations this month",
    value: "184",
    detail: "Unlimited on your plan",
    percent: null as number | null,
  },
  {
    label: "Concurrent generations",
    value: "1 of 3",
    detail: "Fair-use limit keeps the queue fast for everyone",
    percent: 33,
  },
  {
    label: "Media library",
    value: "2.4 GB",
    detail: "of 100 GB included",
    percent: 24,
  },
] as const;

export const mockBilling = {
  status: "Active",
  renewsOn: "9 September 2026",
  method: "Visa ending 4242",
  methodDetail: "Expires 08 / 2029",
  billingEmail: "maya@zolexai.com",
} as const;

export interface MockInvoice {
  id: string;
  date: string;
  amount: string;
  status: "Paid";
}

export const mockInvoices: MockInvoice[] = [
  { id: "ZX-2026-0008", date: "9 August 2026", amount: "$70.00", status: "Paid" },
  { id: "ZX-2026-0007", date: "9 July 2026", amount: "$70.00", status: "Paid" },
  { id: "ZX-2026-0006", date: "9 June 2026", amount: "$70.00", status: "Paid" },
  { id: "ZX-2026-0005", date: "9 May 2026", amount: "$70.00", status: "Paid" },
  { id: "ZX-2026-0004", date: "9 April 2026", amount: "$70.00", status: "Paid" },
];
