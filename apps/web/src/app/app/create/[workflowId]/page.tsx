import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { CreatorWorkspace } from "@/components/generation/CreatorWorkspace";
import { loadWorkflow, loadWorkflowCatalog } from "@/features/workflows/catalog.server";

/**
 * ONE dynamic creator route for all six workflows.
 *
 * Non-negotiable rule #6: no per-workflow pages. Adding a workflow means adding
 * a YAML definition — this file never changes.
 *
 * The workflow is resolved on the server from the definitions so the page has
 * its title and a fully-rendered settings panel immediately; the client then
 * reads the same workflow from the API through React Query.
 */

export async function generateStaticParams() {
  const workflows = await loadWorkflowCatalog();
  return workflows.map((workflow) => ({ workflowId: workflow.id }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ workflowId: string }>;
}): Promise<Metadata> {
  const { workflowId } = await params;
  const workflow = await loadWorkflow(workflowId);
  return { title: workflow?.name ?? "Create" };
}

export default async function CreatorWorkspacePage({
  params,
  searchParams,
}: {
  params: Promise<{ workflowId: string }>;
  searchParams: Promise<{ prompt?: string; source?: string }>;
}) {
  const { workflowId } = await params;
  const workflow = await loadWorkflow(workflowId);

  // An unknown workflow id is a 404, not a silent fallback — otherwise a typo
  // would quietly render Text to Video and look like it worked.
  if (!workflow) notFound();

  // Read here rather than with `useSearchParams()` in the client component.
  // That hook forces a client-side bailout during prerendering and needs a
  // Suspense boundary around the entire workspace; passing the value down as a
  // prop keeps the workspace a single, fully server-rendered tree.
  //
  // `source` is how Extend hands a finished result to its own tool. It is only
  // an asset id — ownership is enforced by the API when the input is resolved
  // and again when the generation is created, so a guessed id gains nothing.
  const { prompt, source } = await searchParams;

  return (
    <CreatorWorkspace
      workflowId={workflowId}
      initialWorkflow={workflow}
      initialPrompt={prompt ?? ""}
      initialSourceAssetId={source ?? null}
    />
  );
}
