import type { Metadata } from "next";
import { GenerationDetail } from "@/components/generation/GenerationDetail";

export const metadata: Metadata = { title: "Generation" };

/**
 * Generation detail.
 *
 * No `generateStaticParams` — and there cannot be one. Generation ids exist
 * only at runtime, per user. The PRE-M1 version prerendered a fixed list of
 * mock ids, which is precisely the assumption that made static export possible
 * and is no longer true.
 */
export default async function GenerationDetailPage({
  params,
}: {
  params: Promise<{ generationId: string }>;
}) {
  const { generationId } = await params;
  return <GenerationDetail generationId={generationId} />;
}
