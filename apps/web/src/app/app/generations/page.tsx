import type { Metadata } from "next";
import { GenerationsView } from "@/components/generation/GenerationsView";

export const metadata: Metadata = { title: "Generations" };

export default function GenerationsPage() {
  return <GenerationsView />;
}
