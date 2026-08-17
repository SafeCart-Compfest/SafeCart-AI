# Evaluation dataset

## Review scope

Reviewers determine whether the identity represented by one marketplace listing is
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
   information before sharing outside the private review workspace.
2. Record visible text without correcting it from the official record.
3. Retrieve every official candidate for the observed NIE. Do not silently choose one
   when the official snapshot is ambiguous.
4. Assign one label and one or more controlled reason codes.
5. Two reviewers work independently. They must not see each other's initial label.
6. Resolve every disagreement and preserve both initial labels plus the resolved label.
7. Finalize the 120-sample test set before selecting model thresholds.

Use `review_stage=initial` for the two independent reviews and
`review_stage=resolution` for the optional resolution row. Separate multiple reason
codes with semicolons. `readability` must be `READABLE`, `PARTIALLY_READABLE`, or
`UNREADABLE`. Allowed reason codes are:

- `IDENTITY_CONSISTENT`
- `NIE_MISMATCH`, `NIE_NOT_FOUND`, or `OFFICIAL_NIE_AMBIGUOUS`
- `BRAND_MISMATCH`, `PRODUCT_NAME_MISMATCH`, or `VARIANT_MISMATCH`
- `PACKAGE_MISMATCH`, `SPF_MISMATCH`, `SHADE_MISMATCH`, or `STRENGTH_MISMATCH`
- `UNREADABLE`, `MISSING_IDENTITY_FIELDS`, `MULTIPLE_PLAUSIBLE_RECORDS`, or
  `SOURCE_UNVERIFIABLE`

Validate work in progress with:

```bash
safecart-ai-validate-evaluation-dataset private/evaluation-dataset.csv
```

The finalized dataset must additionally pass:

```bash
safecart-ai-validate-evaluation-dataset private/evaluation-dataset.csv --final
```

## Target composition

- 50 identity-consistent listings.
- 50 verified mismatches covering subtle variants, package, SPF, shade, and NIE changes.
- 20 ambiguous, unreadable, or otherwise insufficient-evidence listings.

Never infer `MISMATCH` from folder names such as `reported_counterfeit_candidate`.
