# Internal engineering notes

**Not customer-facing.** Everything in this directory may name models, providers,
hardware and vendors. Nothing here should be reproduced in the product UI, the
public API, marketing copy, or a document sent to an end user.

That split is a product rule, not a habit: the public surface is deliberately
provider-agnostic so a model can be replaced without a customer noticing, and
`apps/api/tests/test_workflows.py` fails if a provider name reaches a public
response. These notes are where the other half lives.

| File | Subject |
|---|---|
| [`ltx-2.5-licensing-review.md`](./ltx-2.5-licensing-review.md) | Licence and commercial-use review for the candidate video model |
| [`production-runbook.md`](./production-runbook.md) | Production server layout, deploy/rollback procedure, go-live checks |

The runbook names the VPS, its addresses, the deployment paths and the
CloudPanel routing, which is why it lives here rather than beside the
client-facing control documents. It carries **no secrets** by design — no
passwords, keys, tokens or `.env` values — and it must stay that way: the
production `.env` is on the server at mode 600 and belongs nowhere else.

It documents the state as deployed for **M1 (mock worker)**. The M2 GPU runtime
is not deployed and is not described there; when it is, that section is written
at deploy time from what was actually done, not in advance.
