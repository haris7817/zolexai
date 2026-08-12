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
