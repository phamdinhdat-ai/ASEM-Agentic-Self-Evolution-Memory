# CARE claim-level source-support audit

The full LoCoMo results contain 1,986 CARE predictions. A deterministic
SHA-256 selection freezes 100 non-abstaining answers: ten for each pairing
of the five LoCoMo categories with one-pass versus repaired output. This
balanced sample deliberately overrepresents rare categories and repaired
answers. It is suited to failure analysis, not an unweighted population
estimate of source support.

The JSON and spreadsheet-ready CSV annotation queues contain the question, final answer, and only the
original turns named by the answer's cited source IDs, with speaker and
session date. It omits the benchmark reference, LoCoMo F1 score, annotated
gold evidence, and repair status. These are stored in a separate key file.
The exporter verifies that every citation exists and was visible to CARE.

An independent annotator should label each answer as:

- `supported`: the cited turns jointly entail the answer without relying on
  uncited conversation content;
- `partial`: at least one answer claim is supported but another is not;
- `unsupported`: the cited turns do not establish the answer;
- `uncertain`: the cited text is ambiguous or the annotation cannot be made.

The annotator separately marks whether speaker attribution and time are
consistent with the citation (`yes`, `no`, `uncertain`) and records a short
reason. A second annotator should independently label at least 20 items
before disagreements are adjudicated. A future report should give raw
counts by category and repair status, agreement before adjudication, and
examples of unsupported answers. If estimating a population-wide rate,
weight each stratum by its frequency among all non-abstaining predictions.
No support-rate claim is made until labels exist.
