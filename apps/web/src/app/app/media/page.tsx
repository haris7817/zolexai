import type { Metadata } from "next";
import { MediaLibraryView } from "@/components/media/MediaLibraryView";

export const metadata: Metadata = { title: "Media Library" };

export default function MediaLibraryPage() {
  return <MediaLibraryView />;
}
