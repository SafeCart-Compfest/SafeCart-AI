# Retrieval baseline

Official-record retrieval uses a transparent union rather than one opaque similarity
score:

1. return every exact normalized NIE candidate;
2. block lexical candidates by normalized brand and informative product-name tokens;
3. fit character TF-IDF n-grams (3-5) once on the canonical catalog;
4. rank at most 5,000 blocked records with TF-IDF similarity and RapidFuzz;
5. preserve retrieval sources and deterministic record-ID tie breaking.

Evaluation reports Recall@1, Recall@5, Recall@20, and mean reciprocal rank on positive
pairs. Run both normal mode and `--lexical-only`; the latter removes NIE to prevent an
exact lookup from hiding weak name retrieval. Synthetic metrics are baselines, not the
headline gold-test result.

Every metrics file records input hashes, index/query timing, and deterministic
percentile-bootstrap 95% confidence intervals. Use `--output` to persist a run outside
Git before copying reviewed aggregate results into an experiment report.
