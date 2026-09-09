# Vast.ai + llama.cpp for Zero-Shot LJP

This guide runs the zero-shot Legal Judgment Prediction workflow with
`unsloth/Qwen3.5-9B-GGUF` served by llama.cpp on a Vast.ai GPU instance.

The LJP workflow does not call the ALQAC Case Content API. It uses:

```text
case_fact -> input processing -> law retrieval -> judgment reasoning -> prediction
```

The output file contains only:

```json
[
  {"case_id": "case_4101", "prediction": "PARTIAL_A_WIN"}
]
```

## 1. Rent a Vast.ai instance

Recommended hardware:

```text
GPU VRAM: 16GB minimum, 24GB recommended
Disk: 80GB or more
Image: Ubuntu 22.04 with CUDA
```

The model card recommends llama.cpp usage with:

```bash
llama serve -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL
```

## 2. Install llama.cpp

SSH into the Vast.ai instance, then run:

```bash
apt-get update
apt-get install -y git curl tmux python3 python3-venv python3-pip
curl -LsSf https://llama.app/install.sh | sh
source ~/.bashrc
llama --version
```

## 3. Start the Qwen server

Start a persistent terminal:

```bash
tmux new -s qwen
```

Run llama.cpp:

```bash
~/.llama-app/llama serve \
  -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M \
  --no-mmproj \
  --reasoning off \
  --host 0.0.0.0 \
  --port 8000 \
  -ngl all \
  -c 16384 \
  -np 1 \
  -a qwen3.5-9b
```

Keep this server running. Detach from tmux with `Ctrl-b`, then `d`.

If your llama.cpp build does not support `--reasoning off`, replace it with
`--reasoning-budget 0`. If neither flag exists, keep `LLM_MAX_TOKENS` modest
when running experiments.

Smoke-test the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Return JSON only: {\"ok\": true}"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## 4. Install this repository

Open another terminal on the same Vast.ai instance:

```bash
cd /workspace
git clone <repo-url> AgentUIT_at_ALQAC2026
cd AgentUIT_at_ALQAC2026

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 5. Run zero-shot LJP

Smoke test one case:

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

Run all public cases:

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

Evaluate on public labels:

```bash
alqac-agent evaluate \
  --gold ALQAC2026_public_test.json \
  --predictions runs/submission.ljp.public.json
```

## 6. Run paper experiments

Use the experiment runner when collecting paper results. It writes final outputs,
metrics, traces, confusion matrices, qualitative files, and summary tables under
`result/`.

Prompt-only baseline:

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
  --runs 3 \
  --no-resume
```

Proposed method:

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
  --runs 3 \
  --no-resume
```

With `--runs 3`, each run is saved separately using `_run_01`, `_run_02`, and
`_run_03` suffixes. The summary tables report only the mean of the three runs.

Use the same command with these modes for ablation:

```text
no_law_retrieval
no_input_processing
no_law_retrieval_no_input_processing
```

The summary tables are refreshed after every experiment:

```text
result/tables/main_comparison.md
result/tables/ablation_study.md
```

## 7. Run without LLM

This ablation uses only input-processing rules, BM25 law retrieval, and the
deterministic resolver:

```bash
alqac-agent run-ljp \
  --input ALQAC2026_public_test.json \
  --laws corpus_law_pub.json \
  --output runs/submission.ljp.public.no_llm.json \
  --trace-output runs/submission.ljp.public.no_llm.trace.json \
  --no-llm \
  --no-resume
```
