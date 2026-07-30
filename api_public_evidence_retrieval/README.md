# API Public Evidence Retrieval

This folder is a standalone package for the public-test case-evidence retrieval task.
It contains the optimized query blueprints, public test data, keyword corpus, cached LLM classification/extraction outputs, and the code needed to instantiate queries and call the official Case Content API.

## What Is Included

- `config.json`: internal paths and runtime settings for this package.
- `data/ALQAC2026_public_test.json`: public test cases used by the retrieval pipeline.
- `data/keyword_corpora_by_case_type.simple.json`: keyword corpus by case type.
- `outputs/llm_case_type_classification/`: classifier outputs already generated.
- `outputs/llm_case_query_keywords/`: case-query keyword extraction outputs already generated.
- `blueprints/best_query_blueprints.json`: best optimized query blueprints by case type.
- `optimizer_outputs/evolution_query_optimizer/`: copied optimizer outputs used to rebuild the blueprint registry.
- `src/llm_case_type_classifier.py`: reusable LLM case-type classifier code.
- `src/llm_case_query_keyword_extractor.py`: reusable LLM keyword extractor code.
- `retrieve_public_evidence.py`: public-test API retrieval runner.
- `build_blueprint_registry.py`: rebuilds the blueprint registry from optimizer outputs.

The blueprint registry currently includes all 11 configured case types.

## Setup

Create a real `.env` file inside this folder from `.env.example`:

```powershell
Copy-Item api_public_evidence_retrieval\.env.example api_public_evidence_retrieval\.env
```

Then set:

```text
ALQAC_TOKEN=your_team_secret_token
```

The real `.env` file is intentionally not included for packaging.

## Run Without Rerunning LLM

The retrieval pipeline can run immediately from the copied classifier and extractor outputs. It does not need Qwen unless you want to regenerate those outputs.

Dry run for one case, without calling the API:

```powershell
.\.venv\Scripts\python.exe api_public_evidence_retrieval\retrieve_public_evidence.py --case-id case_4588 --dry-run --overwrite
```

Retrieve one case through the API:

```powershell
.\.venv\Scripts\python.exe api_public_evidence_retrieval\retrieve_public_evidence.py --case-id case_4588 --overwrite
```

Or use the dedicated one-case runner:

```powershell
.\.venv\Scripts\python.exe api_public_evidence_retrieval\retrieve_one_case.py case_4588
```

Dry run the one-case runner:

```powershell
.\.venv\Scripts\python.exe api_public_evidence_retrieval\retrieve_one_case.py case_4588 --dry-run
```

The one-case runner saves to:

```text
api_public_evidence_retrieval/outputs/per_case/<case_id>_api_evidence.jsonl
```

Retrieve public test in small batches:

```powershell
.\.venv\Scripts\python.exe api_public_evidence_retrieval\retrieve_public_evidence.py --start 0 --limit 5 --overwrite
```

Default output:

```text
api_public_evidence_retrieval/outputs/public_api_evidence_retrieval.jsonl
```

API responses are cached at:

```text
api_public_evidence_retrieval/.cache/case_retrieval_cache.json
```

## Rebuild Blueprint Registry

```powershell
.\.venv\Scripts\python.exe api_public_evidence_retrieval\build_blueprint_registry.py
```

Rebuild the registry after copying in newer optimizer outputs:

```powershell
.\.venv\Scripts\python.exe api_public_evidence_retrieval\build_blueprint_registry.py
```

## Rerun Classifier Or Extractor

Only do this if you want to regenerate LLM outputs. Install the LLM dependencies:

```powershell
.\.venv\Scripts\pip.exe install -r api_public_evidence_retrieval\requirements-llm.txt
```

Place the Qwen GGUF model here, or pass `--model` manually:

```text
api_public_evidence_retrieval/models/qwen2.5-3b-instruct-gguf/Qwen2.5-3B-Instruct-Q4_K_M.gguf
```

Run classifier for all public cases:

```powershell
.\.venv\Scripts\python.exe api_public_evidence_retrieval\src\run_public_llm_case_type_classifier.py --resume
```

Run keyword extractor for all public cases:

```powershell
.\.venv\Scripts\python.exe api_public_evidence_retrieval\src\llm_case_query_keyword_extractor.py batch --split public --resume
```

Test keyword extraction for one public case:

```powershell
.\.venv\Scripts\python.exe api_public_evidence_retrieval\src\test_public_case_query_keyword_extractor.py --case-id case_4588
```

## Output Format

Each JSONL row contains:

```json
{
  "case_id": "...",
  "case_query": "...",
  "case_type_id": "...",
  "case_type_text": "...",
  "keyword_source": "...",
  "num_instantiated_queries": 0,
  "api_calls_made": 0,
  "cache_hits": 0,
  "chunks": [
    {
      "chunk_id": "...",
      "text": "...",
      "score": 0.0,
      "source_queries": ["..."]
    }
  ]
}
```
