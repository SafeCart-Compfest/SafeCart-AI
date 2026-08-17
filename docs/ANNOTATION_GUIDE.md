# Gold evaluation annotation guide

## Claim being labeled

Annotators judge whether the identity represented by one marketplace listing is
consistent with a specific official BPOM record. They do not judge physical authenticity,
chemical safety, seller intent, or legal liability.

## Labels

- `MATCH`: readable listing evidence consistently identifies the official product.
- `MISMATCH`: readable evidence contradicts the official record in NIE, brand, product,
  variant, package, strength, SPF, shade, or another identity-bearing attribute.
- `INSUFFICIENT_EVIDENCE`: the listing is unreadable, missing identity-bearing fields, the
  source cannot be verified, or multiple official records remain plausible.

## Procedure

1. Capture the listing URL, timestamp, and a private screenshot; redact seller personal
   information before sharing outside the annotation workspace.
2. Record visible text without correcting it from the official record.
3. Retrieve every official candidate for the observed NIE. Do not silently choose one
   when the official snapshot is ambiguous.
4. Assign one label and one or more controlled reason codes.
5. Two annotators work independently. They must not see each other's first-round label.
6. Adjudicate every disagreement and preserve both original labels plus the final label.
7. Freeze the 120-sample test set before selecting model thresholds.

## Target composition

- 50 identity-consistent listings.
- 50 verified mismatches covering subtle variants, package, SPF, shade, and NIE changes.
- 20 ambiguous, unreadable, or otherwise insufficient-evidence listings.

Never infer `MISMATCH` from folder names such as `reported_counterfeit_candidate`.
