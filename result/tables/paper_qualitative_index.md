# Paper Qualitative Analysis Index

This file lists compact qualitative evidence to inspect when writing the paper. The raw qualitative artifacts remain in `result/qualitative/`.

## Input Processing Module

Recommended source file:

- `result/qualitative/qwen3_5_9b_9b_method_run_01.input_processing_outputs.json`

Recommended examples:

- `case_4101`: useful for showing extraction of the plaintiff's compensation claim, requested amounts, and a court disposition candidate indicating partial acceptance.
- `case_4337`: useful for showing how land/house dispute claims are condensed before law retrieval and outcome reasoning.

What to discuss:

- The module extracts claim-oriented text from long `case_fact`.
- It identifies money amounts and candidate court/procuracy disposition spans.
- It reduces the input from full narrative form to decision-relevant snippets.

## Law Retrieval Module

Recommended source file:

- `result/qualitative/qwen3_5_9b_9b_method_run_01.retrieval_outputs.json`

Recommended examples:

- `case_4337`: good retrieval example for a land-use dispute; retrieved candidates include Land Law provisions.
- `case_4101`: useful failure/limitation example; the query is claim-relevant, but some top retrieved laws are only weakly related to animal-caused damage and civil compensation.

What to discuss:

- Retrieval is zero-shot BM25-style over `corpus_law_pub.json`.
- Relevant legal terms in the processed query can guide retrieval toward proper domains.
- Retrieval noise remains a limitation, especially when facts contain many generic litigation terms.

## Special Cases And Error Analysis

Recommended source file:

- `result/qualitative/qwen3_5_9b_9b_method_run_01.special_cases.json`

Recommended examples:

- `case_4101`: `partial_label_boundary`; good example of partial acceptance where the model must distinguish full vs partial plaintiff success.
- Inspect cases marked in this file with wrong predictions to discuss boundary errors among `A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, and `B_WIN`.

What to discuss:

- The hardest class remains `PARTIAL_B_WIN`, which many open-source models fail to predict correctly.
- Boundary labels are difficult because the model must compare requested relief with granted and rejected relief.
- Structured input processing helps, but law retrieval alone is insufficient when the final disposition is ambiguous.

## Main Quantitative Files For Paper

- `result/tables/paper_main_comparison.md`
- `result/tables/paper_main_comparison.csv`
- `result/tables/paper_ablation_study.md`
- `result/tables/paper_ablation_study.csv`

## Raw Tables Kept For Reproducibility

- `result/tables/main_comparison.md`
- `result/tables/main_comparison.csv`
- `result/tables/ablation_study.md`
- `result/tables/ablation_study.csv`

