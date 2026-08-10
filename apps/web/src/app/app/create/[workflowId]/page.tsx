import type { Metadata } from "next";
import { notFound } from "next/navigation";
import {
  WORKFLOW_ORDER,
  getWorkflow,
} from "@/features/workflows/registry";
import { CreatorWorkspace } from "@/components/generation/CreatorWorkspace";

/**
 * ONE dynamic creator route for all six workflows.
 *
 * Architecture doc §41 and non-negotiable rule #6: no per-workflow pages.
 * Adding a workflow in M2 means adding a registry entry — this file never
 * changes.
 */

export function generateStaticParams() {
  return WORKFLOW_ORDER.map((workflowId) => ({ workflowId }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ workflowId: string }>;
}): Promise<Metadata> {
  const { workflowId } = await params;
  const workflow = getWorkflow(workflowId);
  return { title: workflow?.name ?? "Create" };
}

export default async function CreatorWorkspacePage({
  params,
}: {
  params: Promise<{ workflowId: string }>;
}) {
  const { workflowId } = await params;
  const workflow = getWorkflow(workflowId);

  // An unknown workflow id is a 404, not a silent fallback — otherwise a typo
  // would quietly render Text to Video and look like it worked.
  if (!workflow) notFound();

  return <CreatorWorkspace workflow={workflow} />;
}
