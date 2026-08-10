import type { Metadata } from "next";
import { AllToolsView } from "@/components/workflow/AllToolsView";

export const metadata: Metadata = { title: "All Tools" };

export default function AllToolsPage() {
  return <AllToolsView />;
}
