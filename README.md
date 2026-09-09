# AgentUIT at ALQAC 2026

This repository contains the source code and reproducibility artifacts for the
AgentUIT system submitted to **ALQAC 2026**. The system predicts the judgment
outcome of Vietnamese civil cases and returns supporting case evidence and law
articles.

The repository is designed to let readers reproduce the experiments reported in
our paper:

1. regenerate public-test evidence retrieval query blueprints with an
   evolutionary algorithm;
2. rebuild the blueprint registry used by the retrieval pipeline;
3. retrieve evidence from the ALQAC Case Content API, or use the provided
   pre-retrieved public evidence;
4. run the full grounded LangGraph agent;
5. evaluate public-test predictions.

No model checkpoint is committed. The full agent uses `Qwen/Qwen3-8B` by
default. The optional case-type classifier and keyword extractor use
`Qwen/Qwen2.5-3B-Instruct-GGUF` with the `q4_k_m` GGUF file and download it by
name through the Hugging Face cache when needed.

## Repository Layout

```text
.
|-- src/alqac_agent/                    # Full prediction agent and CLI
|-- tests/                              # Offline regression/unit tests
|-- api_public_evidence_retrieval/      # Public evidence retrieval pipeline
|   |-- blueprints/                     # Best query-blueprint registry
|   |-- optimizer_outputs/              # Evolutionary optimizer outputs
|   |-- outputs/                        # Cached classifier/keyword/evidence outputs
|   |-- prompts/                        # LLM prompts for retrieval submodules
|   `-- src/                            # Case type and keyword LLM modules
|-- tools/evolution_query_optimizer.py  # Evolutionary query optimizer
|-- configs/evolution_query_optimizer.json
|-- ALQAC2026_public_test.json          # Public test set with labels
|-- ALQAC2026_public_input.json         # Label-blind public input
|-- ALQAC_private_test.json             # Private input format
|-- corpus_law_pub.json                 # Law article corpus
|-- public_test_corpus.jsonl            # Public local case chunks
|-- pyproject.toml
`-- TECHNICAL_REPORT.md
```

Ignored or intentionally omitted files include `.env`, `.cache/`,
`.pytest_cache/`, `__pycache__/`, `*.egg-info/`, intermediate submissions, and
all model files.

## Environment

Python 3.11 or newer is recommended.

```bash
git clone <repo-url>
cd AgentUIT_at_ALQAC2026

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows PowerShell:

```powershell
git clone <repo-url>
cd AgentUIT_at_ALQAC2026

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the offline checks:

```bash
alqac-agent validate \
  --input ALQAC_private_test.json \
  --laws corpus_law_pub.json \
  --examples ALQAC2026_public_test.json

alqac-agent inspect-law "boi thuong thiet hai do suc vat gay ra"
pytest
```

## Model

No model file is stored in this repository. Use the model name below; the serving
runtime will download/load it by name when needed:

```text
Qwen/Qwen3-8B
```

Set `LLM_MODEL=Qwen/Qwen3-8B` if you override the default environment.

The optional retrieval feature regeneration scripts use a smaller local GGUF
model:

```text
Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q4_k_m.gguf
```

When the default local file is absent, the scripts download this file
automatically through `huggingface_hub`.

## API Credentials

For official Case Content API retrieval:

```bash
export ALQAC_API_KEY=<your-alqac-api-key>
```

For the standalone public-evidence retrieval package:

```bash
cp api_public_evidence_retrieval/.env.example api_public_evidence_retrieval/.env
```

Then edit:

```text
ALQAC_TOKEN=<your-alqac-api-token>
```

The two names are historical: `ALQAC_API_KEY` is used by the full agent, while
`ALQAC_TOKEN` is used by `api_public_evidence_retrieval`.

## Reproduce Query Blueprint Optimization

The evolutionary optimizer searches query blueprints for each case type. A
blueprint is a list of slots such as `name1`, `number1`, or `corpus:23`; at
runtime these slots are instantiated into concrete API queries for each case.

The committed optimizer outputs are already available under:

```text
api_public_evidence_retrieval/optimizer_outputs/evolution_query_optimizer/
```

To rerun a quick smoke optimization for one case type:

```bash
python tools/evolution_query_optimizer.py \
  --config configs/evolution_query_optimizer.json \
  run \
  --case-type-id loan_contract \
  --population-size 20 \
  --generations 3 \
  --limit-cases 3 \
  --output-dir api_public_evidence_retrieval/optimizer_outputs/evolution_query_optimizer_smoke
```

To rerun the full optimization used for the paper artifacts:

```bash
python tools/evolution_query_optimizer.py \
  --config configs/evolution_query_optimizer.json \
  run-all \
  --output-dir api_public_evidence_retrieval/optimizer_outputs/evolution_query_optimizer
```

This uses the settings in `configs/evolution_query_optimizer.json`:
`population_size=200`, `generations=300`, and early stopping at fitness `0.97`.
The full run may take a long time.

After generating optimizer outputs, rebuild the registry consumed by the public
API retriever:

```bash
python api_public_evidence_retrieval/build_blueprint_registry.py
```

The registry is written to:

```text
api_public_evidence_retrieval/blueprints/best_query_blueprints.json
```

## Optional: Regenerate LLM Retrieval Features

The repository includes cached public case-type predictions and keyword
extraction outputs, so this step is optional. Run it only when changing the local
model or prompts. Both scripts use
`Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q4_k_m.gguf` by default and
download it automatically if it is not already in the Hugging Face cache.

```bash
python api_public_evidence_retrieval/src/run_public_llm_case_type_classifier.py --resume

python api_public_evidence_retrieval/src/llm_case_query_keyword_extractor.py \
  batch \
  --split public \
  --resume
```

Smoke test one case:

```bash
python api_public_evidence_retrieval/src/test_public_case_query_keyword_extractor.py \
  --case-id case_4588 \
  --no-save
```

## Retrieve Public Evidence

Dry-run one case without calling the official API:

```bash
python api_public_evidence_retrieval/retrieve_public_evidence.py \
  --case-id case_4588 \
  --dry-run \
  --overwrite
```

Retrieve one case through the official API:

```bash
python api_public_evidence_retrieval/retrieve_public_evidence.py \
  --case-id case_4588 \
  --overwrite
```

Retrieve the full public set:

```bash
python api_public_evidence_retrieval/retrieve_public_evidence.py --overwrite
```

Default output:

```text
api_public_evidence_retrieval/outputs/public_api_evidence_retrieval.jsonl
```

The retriever caches API responses in:

```text
api_public_evidence_retrieval/.cache/case_retrieval_cache.json
```

## Run the Full Agent

First prepare label-blind public input if you want to recreate it from the public
test file:

```bash
alqac-agent prepare-public \
  --source ALQAC2026_public_test.json \
  --output ALQAC2026_public_input.json
```

### Public Run with Local Public Backend

This is fully offline except for the configured LLM endpoint. It simulates the
case evidence API using BM25 over `judgment_text` chunks from the public set.

```bash
alqac-agent run \
  --input ALQAC2026_public_input.json \
  --laws corpus_law_pub.json \
  --case-backend local-public \
  --case-corpus ALQAC2026_public_test.json \
  --output runs/submission.public.local.json \
  --no-resume
```

Evaluate:

```bash
alqac-agent evaluate \
  --gold ALQAC2026_public_test.json \
  --predictions runs/submission.public.local.json
```

### Public Run with Pre-Retrieved Evidence

This uses `api_public_evidence_retrieval/outputs/public_api_evidence_retrieval.jsonl`
and does not call the Case Content API.

```bash
alqac-agent run \
  --input ALQAC2026_public_input.json \
  --laws corpus_law_pub.json \
  --case-backend preloaded \
  --evidence-jsonl api_public_evidence_retrieval/outputs/public_api_evidence_retrieval.jsonl \
  --output runs/submission.public.preloaded.json \
  --no-resume
```

Evaluate:

```bash
alqac-agent evaluate \
  --gold ALQAC2026_public_test.json \
  --predictions runs/submission.public.preloaded.json
```

### Official API Run

Use this mode for the official private-style setting.

```bash
export ALQAC_QUERY_MODE=fact
export ALQAC_INITIAL_CASE_QUERIES=1
export ALQAC_MAX_CASE_QUERIES=1
export ALQAC_API_KEY=<your-alqac-api-key>

alqac-agent run \
  --input ALQAC_private_test.json \
  --laws corpus_law_pub.json \
  --case-backend official \
  --output runs/submission.private.json \
  --no-resume
```

The agent writes:

```text
runs/submission.private.json
runs/submission.private.json.evidence.json
```

The first file follows the ALQAC submission schema. The second file stores a
development trace with prompts, retrieved chunks, decisions, and law candidates.

## Run Legal Judgment Prediction From Case Facts

This workflow is separate from the original ALQAC evidence-retrieval graph. It
does not call the ALQAC Case Content API and does not retrieve case evidence.
The input is the `case_fact` field from `ALQAC2026_public_test.json`; the graph
builds local case-fact snippets, retrieves law articles from `corpus_law_pub.json`
with the same BM25 law retriever, extracts the disposition, and predicts the
judgment label. The output contains only `case_id` and `prediction`; internal
snippets, law query, retrieved laws, and reasoning details can be saved with
`--trace-output`.

Run a smoke test:

```bash
alqac-agent run-ljp \
  --input ALQAC2026_public_test.json \
  --laws corpus_law_pub.json \
  --output runs/submission.ljp.public.limit1.json \
  --trace-output runs/submission.ljp.public.limit1.trace.json \
  --limit 1 \
  --model qwen3.5-9b \
  --llm-provider vllm \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --structured-method prompt_json \
  --no-resume
```

Evaluate:

```bash
alqac-agent evaluate \
  --gold ALQAC2026_public_test.json \
  --predictions runs/submission.ljp.public.limit1.json
```

For a llama.cpp server on Vast.ai, start a GGUF model with an OpenAI-compatible
endpoint:

```bash
llama serve \
  -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 \
  --port 8000 \
  -ngl all \
  -c 16384 \
  -np 1 \
  -a qwen3.5-9b
```

Then run the full case-fact LJP graph:

```bash
alqac-agent run-ljp \
  --input ALQAC2026_public_test.json \
  --laws corpus_law_pub.json \
  --output runs/submission.ljp.public.json \
  --trace-output runs/submission.ljp.public.trace.json \
  --model qwen3.5-9b \
  --llm-provider vllm \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --structured-method prompt_json \
  --no-resume
```

For an ablation without any LLM server, add `--no-llm`; this uses only local
case-fact cues, BM25 law retrieval, and deterministic outcome rules.

### Experiment Runner

Use `run-ljp-experiment` for paper experiments. This command writes every final
and intermediate output into `result/` by type:

```text
result/submissions/          # final case_id + prediction outputs
result/metrics/              # accuracy, macro-F1, per-label scores, coverage
result/confusion_matrices/   # confusion matrices split out for paper figures
result/traces/               # processed input, retrieved laws, reasoning trace
result/qualitative/          # retrieval/input/special-case qualitative files
result/tables/               # main comparison and ablation tables
```

Prompt-only baseline for a served model:

```bash
alqac-agent run-ljp-experiment \
  --input ALQAC2026_public_test.json \
  --laws corpus_law_pub.json \
  --result-dir result \
  --type "General Domain" \
  --backbone "Qwen3.5-9B" \
  --params "9B" \
  --domain "General" \
  --mode prompt_only \
  --model qwen3.5-9b \
  --llm-provider vllm \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --structured-method prompt_json \
  --no-resume
```

The same model with the proposed method:

```bash
alqac-agent run-ljp-experiment \
  --input ALQAC2026_public_test.json \
  --laws corpus_law_pub.json \
  --result-dir result \
  --type "General Domain" \
  --backbone "Qwen3.5-9B" \
  --params "9B" \
  --domain "General" \
  --mode method \
  --model qwen3.5-9b \
  --llm-provider vllm \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --structured-method prompt_json \
  --no-resume
```

Ablation modes:

```text
no_law_retrieval
no_input_processing
no_law_retrieval_no_input_processing
```

The summary tables are refreshed after each run:

```text
result/tables/main_comparison.csv
result/tables/main_comparison.md
result/tables/ablation_study.csv
result/tables/ablation_study.md
```

See [docs/vastai_llamacpp_ljp.md](docs/vastai_llamacpp_ljp.md) for the full
Vast.ai setup and run guide.

## Main Configuration Variables

```text
LLM_PROVIDER=vllm|ollama|groq
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=Qwen/Qwen3-8B
LLM_API_KEY=EMPTY
LLM_STRUCTURED_METHOD=json_schema|prompt_json

ALQAC_API_URL=https://alqac-api.ngrok.pro/retrieve
ALQAC_API_KEY=<official-api-key>
ALQAC_QUERY_MODE=fact|minimal|disposition
ALQAC_INITIAL_CASE_QUERIES=1
ALQAC_MAX_CASE_QUERIES=1
ALQAC_LAW_CANDIDATES=18
ALQAC_MAX_LAW_EVIDENCE=8
```

See `.env.example` for the complete list.

## Citation

If this repository helps your work, please cite the corresponding ALQAC 2026
paper from Team AgentUIT. A BibTeX entry can be added here after publication.

```bibtex
@inproceedings{agentuit-alqac2026,
  title     = {AgentUIT at ALQAC 2026},
  author    = {Team AgentUIT},
  booktitle = {Proceedings of ALQAC 2026},
  year      = {2026}
}
```
