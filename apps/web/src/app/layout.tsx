import type { Metadata, Viewport } from "next";
import { Manrope } from "next/font/google";
import { brand } from "@/config/brand";
import { Providers } from "./providers";
import { loadWorkflowCatalog } from "@/features/workflows/catalog.server";
import "./globals.css";

const manrope = Manrope({
  variable: "--font-manrope",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: `${brand.name} — ${brand.tagline}`,
    template: `${brand.name} — %s`,
  },
  description: brand.description,
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: "#0A0A0B",
  width: "device-width",
  initialScale: 1,
};

/**
 * The catalogue is read from the YAML definitions on the server and handed to
 * the query cache, so navigation and tool grids paint complete on first render
 * instead of as skeletons — and the landing page keeps working even if the API
 * is unreachable. The live query still runs and takes over.
 */
export default async function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const workflows = await loadWorkflowCatalog();

  return (
    <html lang="en" className={manrope.variable}>
      <body className="bg-zx-bg text-zx-text antialiased">
        <Providers initialWorkflows={workflows}>{children}</Providers>
      </body>
    </html>
  );
}
