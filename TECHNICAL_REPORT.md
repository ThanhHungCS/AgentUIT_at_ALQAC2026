# ALQAC 2026 — Legal Case Outcome Prediction: Technical Report

## 1. Overview

This report describes the method used in the ALQAC 2026 legal case outcome prediction pipeline. The system predicts one of four verdict labels (`A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, `B_WIN`) for a given case query by combining retrieved case evidence, statutory law retrieval, BM25-based kNN priors, and LLM-based reasoning.

The pipeline was run on the private test set (60 cases) using the official ALQAC Case Content API for evidence retrieval and a Qwen 3.5-9B model (served via llama.cpp) for structured LLM calls.

## 2. System Architecture

The pipeline is implemented as a LangGraph state machine with 10 nodes, each performing a specific function in the prediction workflow:

```
START → plan → retrieve_initial → retrieve_law_context → assess_disposition
                                                            ↓
                                                    [conditional route]
                                                    ↙               ↘
                                              followup          laws
                                                ↓                 ↓
                                        reassess_disposition      laws
                                                ↓                 ↓
                                              laws ←──────────────┘
                                                ↓
                                        resolve_outcome
                                                ↓
                                        debate_judge
                                                ↓
                                           finalize → END
```

### 2.1 Node Descriptions

| Node | Function | LLM Calls | Description |
|------|----------|-----------|-------------|
| `plan` | Planning | 0–1 | Generates search queries for evidence retrieval. Uses LLM planner for official API backend; skips LLM for preloaded backend (uses fixed disposition probes). |
| `retrieve_initial` | Evidence Retrieval | 0 | Queries the Case Content API (or preloaded JSONL) with disposition-focused probes to retrieve case evidence chunks. |
| `retrieve_law_context` | Law Retrieval | 0 | BM25 search over the law corpus using case query + evidence text. Filters out non-civil laws. |
| `assess_disposition` | Disposition Assessment | 0–1 | First checks for explicit operative-order text (regex-based). If not found, calls LLM to assess court disposition from evidence. |
| `followup` | Follow-up Retrieval | 0 | If disposition is unclear and query budget remains, retrieves additional evidence using follow-up queries. |
| `reassess_disposition` | Re-assessment | 0–1 | Re-runs disposition assessment after follow-up retrieval. |
| `laws` | Law Selection | 1 | LLM selects the most relevant law articles from BM25 candidates. Falls back to top-5 BM25 if LLM fails. |
| `resolve_outcome` | Outcome Resolution | 0 | Combines operative-order parser, disposition assessment, and BM25 kNN prior to produce an initial prediction. |
| `debate_judge` | Judge Refinement | 1 | LLM judge reviews the case with evidence, laws, and examples to potentially override the initial prediction. Skipped if operative-order parser already decided. |
| `finalize` | Finalization | 0 | Assembles the submission payload with prediction, case evidence IDs, and law evidence references. |

## 3. Key Components

### 3.1 Evidence Retrieval

**Backend:** Official ALQAC Case Content API (`https://alqac-api.ngrok.pro/retrieve`)

**Query Strategy:** The system uses fixed disposition-focused probes designed to retrieve the operative portion of court judgments:

```python
DISPOSITION_PROBES = (
    "Vì các lẽ trên QUYẾT ĐỊNH Tuyên xử",
    "Tuyên xử: Chấp nhận Không chấp nhận yêu cầu khởi kiện",
)
```

**Query Budget:**
- Initial queries: 2 (in `minimal` mode)
- Maximum queries: 3 (including follow-up)
- API rate limit: 5.1 seconds between calls

**Supplementary Queries:** The planner may generate additional queries based on extracted monetary amounts or land area claims (e.g., "600.000.000 đồng", "1.949m2") to find relevant financial details in the evidence.

### 3.2 Operative-Order Parser

A deterministic regex-based module that scans retrieved evidence chunks for explicit court disposition text. It uses a scoring function (`disposition_score`) that weights Vietnamese legal phrases:

| Phrase | Weight |
|--------|--------|
| "vì các lẽ trên" | +30 |
| "quyết định" | +25 |
| "tuyên xử" | +18 |
| "chấp nhận một phần yêu cầu" | +15 |
| "không chấp nhận yêu cầu" | +12 |
| "chấp nhận yêu cầu khởi kiện" | +10 |
| "đại diện viện kiểm sát" (without "quyết định") | −20 |

When explicit disposition text is found, the parser maps the court's language to one of four verdict labels:
- **ALL** → `A_WIN` (100% of claims accepted)
- **MAJORITY** → `PARTIAL_A_WIN` (>50% but <100%)
- **MINORITY** → `PARTIAL_B_WIN` (>0% but ≤50%)
- **NONE** → `B_WIN` (0% accepted)

### 3.3 LLM-Based Disposition Assessment

When the operative-order parser does not find explicit text, an LLM call (`extract_disposition`) assesses the evidence. The LLM is prompted to:
1. Identify the court's actual decision (not party statements or prosecutor suggestions)
2. Compare the plaintiff's requested relief with the court's granted relief
3. Classify the claim scope as ALL/MAJORITY/MINORITY/NONE/UNCLEAR
4. Identify decisive evidence chunks
5. Generate follow-up queries if the disposition is unclear

**Input:** Case query + top-4 evidence chunks (text truncated to 500 chars each) + top-5 law candidates (content truncated to 500 chars each).

### 3.4 Law Retrieval and Filtering

**Corpus:** `corpus_law_pub.json` — Vietnamese statutory laws including Civil Code, Civil Procedure Code, Land Law, Credit Law, etc.

**Retrieval:** BM25 index over law article content. Query = case query + plan's law query + evidence text. Top-18 candidates retrieved.

**Non-Civil Law Filtering:** Laws irrelevant to civil disputes are filtered out:

```python
NON_CIVIL_LAW_IDS = {
    "93/2015/QH13",   # Administrative procedure
    "100/2015/QH13",  # Penal code
    "50/2014/QH13",   # Cybersecurity
    "52/2014/QH13",   # Marriage & family
    "52/2010/QH12",   # Adoption
    "39/2009/QH12",   # Elderly
    "02/2011/QH13",   # Complaints
    "60/2014/QH13",   # Civil servants
    "19/2011/NĐ-CP",  # Children
    "24/2012/NĐ-CP",  # Judicial officers
    "66/2014/QH13",   # Enterprise
}
```

If filtering removes too many candidates (<6 remaining), the filter is relaxed.

**LLM Law Selection:** The LLM (`select_laws`) is given the case query, evidence, and up to 10 law candidates (content truncated to 600 chars) and selects only directly relevant law articles. Falls back to top-5 BM25 if LLM fails.

### 3.5 BM25 kNN Outcome Prior

A retrieval-based prior that predicts the verdict label by finding similar cases in the training set:

1. **BM25 Search:** Search for similar case queries in the example set (excluding the current case)
2. **Score Aggregation:** For each verdict label, aggregate BM25 scores weighted by inverse rank: `contribution = score / rank`
3. **Prediction:** The label with the highest aggregated score becomes the prior prediction
4. **Partial Prior:** A separate TF-IDF cosine similarity search restricted to PARTIAL_A_WIN/PARTIAL_B_WIN examples to determine the direction of partial wins

**Parameters:** k=7 neighbors, BM25 over case query text.

### 3.6 Outcome Resolution

The `resolve_outcome` function combines signals in priority order:

1. **Operative-order parser** (highest priority): If explicit disposition text is found, use it directly. Special cases:
   - If parser says PARTIAL_A_WIN but disposition assessment says MINORITY and is grounded → PARTIAL_B_WIN
   - If parser says PARTIAL_A_WIN but partial kNN prior says PARTIAL_B_WIN → PARTIAL_B_WIN
2. **BM25 kNN prior:** If no explicit disposition, use the kNN prior prediction
3. **Grounded disposition assessment:** If disposition is confident (≥0.65), grounded in evidence, and from OPERATIVE_ORDER or COURT_REASONING source → use scope-to-label mapping
4. **Majority fallback:** Default to PARTIAL_A_WIN

**Monetary/Area Overlays:** Post-hoc adjustments based on extracted monetary amounts or land area claims in the case query and evidence.

### 3.7 LLM Judge Refinement

The final LLM step where a judge agent reviews the case:

**Input:**
- Case query
- Top-5 evidence chunks (text truncated to 600 chars each)
- Top-4 law articles (content truncated to 400 chars each)
- Top-3 similar examples (case ID + verdict label only)
- Placeholder advocate opinions (both sides set to current prediction)

**Output:** `JudgeDecision` schema with prediction, confidence, explanation, and selected law evidence.

**Override Logic:**
- If operative-order parser already decided → skip judge
- If judge confidence ≥ 0.7 and judge disagrees with current prediction → override
- Otherwise → keep current prediction, note judge's opinion

### 3.8 Structured Output Method

The system uses `prompt_json` method for LLM structured output:
1. The JSON schema is embedded in the system prompt as instructions
2. The API is called with `response_format={"type": "json_object"}`
3. The response is parsed as JSON and validated against the Pydantic schema
4. Up to 3 retry attempts with error feedback if parsing fails

This method was chosen over `json_schema` (native structured output) because the reasoning model (Qwen 3.5-9B) generates excessive thinking tokens that exhaust the token budget before producing valid JSON in native schema mode.

## 4. LLM Configuration

| Parameter | Value |
|-----------|-------|
| Model | `qwen3.5-9b` (Qwen 3.5, 9B parameters, Q4_K_M quantization) |
| Server | llama.cpp (OpenAI-compatible API) |
| Temperature | 0.0 |
| Max tokens | 16,384 |
| Timeout | 600 seconds |
| Structured output | `prompt_json` |
| Max retries | 2 (API-level), 3 (schema parsing) |

## 5. Run Configuration

| Parameter | Value |
|-----------|-------|
| Case backend | Official ALQAC API |
| Query mode | `minimal` (disposition probes + dynamic queries) |
| Initial case queries | 2 |
| Max case queries | 3 |
| Law candidates | 18 (BM25) |
| Max law evidence | 8 |
| Few-shot examples | 3 |
| kNN neighbors | 7 |
| Disposition confidence threshold | 0.65 |
| API interval | 5.1 seconds |

## 6. Results

### 6.1 Private Test Set (60 cases)

**Prediction Distribution:**

| Label | Count |
|-------|-------|
| A_WIN | 21 |
| PARTIAL_A_WIN | 19 |
| B_WIN | 17 |
| PARTIAL_B_WIN | 3 |

**Decision Source Distribution:**

| Source | Count | Percentage |
|--------|-------|------------|
| BM25 kNN prior + judge confirm | 28 | 46.7% |
| Operative-order parser | 17 | 28.3% |
| Judge override | 12 | 20.0% |
| BM25 kNN prior only | 3 | 5.0% |

**Component Effectiveness:**
- Judge LLM ran successfully: 40/60 (66.7%)
- Disposition found in evidence: 32/60 (53.3%)
- Operative-order parser triggered: 17/60 (28.3%)

### 6.2 Known Limitations

1. **LengthFinishReasonError:** The reasoning model (Qwen 3.5-9B) sometimes generates up to 16,384 thinking tokens without producing valid JSON, causing ~33% of judge calls and ~10% of disposition calls to fail. These cases fall back to the BM25 kNN prior.

2. **API Rate Limiting:** The Case Content API enforces a 5.1-second interval between calls, making evidence retrieval the primary bottleneck (~11 seconds per case for 2-3 queries).

3. **Evidence Quality:** Only 53% of cases have disposition-related content in the retrieved evidence, limiting the effectiveness of the operative-order parser and LLM disposition assessment.

## 7. File Structure

| File | Description |
|------|-------------|
| `src/alqac_agent/graph.py` | LangGraph workflow definition with all pipeline nodes |
| `src/alqac_agent/llm.py` | LLM agent wrappers (plan, disposition, law selection, judge) |
| `src/alqac_agent/outcome.py` | Operative-order parser, disposition scoring, outcome resolution |
| `src/alqac_agent/retrieval.py` | BM25 index, law retriever, example retriever, kNN prior |
| `src/alqac_agent/config.py` | Runtime settings loaded from environment variables |
| `src/alqac_agent/schemas.py` | Pydantic schemas for structured LLM output |
| `src/alqac_agent/runner.py` | Batch runner with evidence trace saving and resume support |
| `src/alqac_agent/cli.py` | Command-line interface |
| `corpus_law_pub.json` | Law corpus (statutory articles) |
| `ALQAC2026_public_test.json` | Public test set with gold labels (used as examples) |
| `ALQAC_private_test.json` | Private test set (60 cases, no labels) |

## 8. Submission Output

The final submission JSON contains:
- `case_id`: Unique case identifier
- `prediction`: One of `A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, `B_WIN`
- `case_evidence`: List of up to 2 chunk IDs from retrieved evidence
- `law_evidence`: List of up to 8 `{law_id, aid}` references to selected law articles

An accompanying evidence trace JSON is also generated for debugging, containing the full retrieved evidence, disposition assessment, outcome prior, decision details, and law candidates for each case.
