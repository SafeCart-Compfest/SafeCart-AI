# Data

## BPOM catalog

The 17 August 2026 snapshot contains 364,133 cosmetic rows and 275,415 unique
normalized NIE values. Catalog construction keeps source metadata and all records when
one NIE maps to multiple product identities. The identity fields are product name,
brand, package, and registrant.

The generated pair dataset uses product-family groups before splitting, which keeps a
product and its generated variants in the same split. Its seed-42 profile contains
826,746 rows with a 1:2 match-to-mismatch ratio and no detected family leakage.

## External data

The supplied medicine image dataset is not used because its product domain and labels
do not support SafeCart's cosmetic listing claim. Images labeled as reported
counterfeit candidates are not treated as verified counterfeit examples.

Raw files remain outside Git. Each committed manifest records its source, snapshot,
permission or license status, preprocessing, and SHA-256 checksum.

## Evaluation examples

Each reviewed example contains listing fields, official candidates, a label, reason
codes, and source metadata. Allowed labels are `MATCH`, `MISMATCH`, and
`INSUFFICIENT_EVIDENCE`. Screenshots remain private and seller information is removed.
