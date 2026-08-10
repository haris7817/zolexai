/**
 * MOCK — PRE-M1 demo only.
 *
 * There is no authentication in this phase. This stand-in user exists so the
 * shell, dashboard and settings screens can be reviewed with realistic content.
 * Real accounts arrive at M3.02.
 */
export const mockUser = {
  name: "Maya Adler",
  email: "maya@zolexai.com",
  initials: "MA",
  plan: "Unlimited",
  memberSince: "March 2026",
  location: "Berlin, Germany",
  timezone: "Europe/Berlin (CET)",
} as const;
