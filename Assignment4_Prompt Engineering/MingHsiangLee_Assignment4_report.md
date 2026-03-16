# Assignment 4: Prompt Engineering for NLP (NER with Local CoNLL-2003)

## Executive Summary
This assignment explores prompt engineering for **Named Entity Recognition (NER)** using local CoNLL-2003 data files.

I implemented and compared multiple prompt formulations (zero-shot and few-shot) across multiple LLMs, then evaluated entity-level extraction performance and output-format reliability.

Notebook:
- `Assignment4_Prompt Engineering/MingHsiangLee_Assignment4.ipynb`

---

## 1. Dataset and Access Method

### Chosen Dataset
- CoNLL-2003 (local files):
1. `eng.train`
2. `eng.testa`
3. `eng.testb`

### Local Relative Path Used in Notebook
The script uses **relative path candidates only**:
- `Assignment2_Name_Entity_Recognition/conll2003`
- `../Assignment2_Name_Entity_Recognition/conll2003`

This avoids hard-coded absolute paths and keeps the notebook portable in the repository.

---

## 2. What Is 0-shot vs Few-shot?

1. **0-shot prompting**
- The prompt contains only task instructions and the input text.
- No labeled examples are given.

2. **Few-shot prompting**
- The prompt contains instructions plus a small set of labeled examples.
- These examples guide the model to follow output style and labeling behavior.

In this assignment, both styles are tested and compared.

---

## 3. Prompt Designs Tested

I implemented four prompt templates:

1. `zero_shot_plain`
   - Basic instruction with explicit JSON format and a note that text may be ALL CAPS.
   - Adds "No explanation. No code." to prevent hallucination.

2. `zero_shot_rubric`
   - Instruction plus per-type definitions: PER, ORG, LOC, MISC with examples.
   - Clarifies ALL-CAPS input. Adds strict "No code. No prose." rule.

3. `few_shot_caps`
   - Three demonstrations using ALL-CAPS text matching CoNLL's style.
   - Examples: cricket teams (ORG), cities (LOC), country match (LOC).

4. `few_shot_balanced`
   - Three demonstrations including multi-entity, no-entity (`[]`), and mixed cases.
   - ALL-CAPS examples reduce domain mismatch vs. the CoNLL input format.

---

## 4. LLM Requirement: Did We Achieve It?

Yes.

The assignment asks to use an LLM, and gives examples such as GPT, Gemini, or LLaMA. These are examples, not strict requirements.

In this notebook, I used instruction-tuned Google FLAN-T5 LLMs:

| Model | Params | Use case |
|---|---:|---|
| `google/flan-t5-small` | 80M | local CPU baseline |
| `google/flan-t5-base` | 250M | local CPU baseline |
| `google/flan-t5-large` *(optional)* | 780M | Colab T4 GPU — recommended |
| `google/flan-t5-xl` *(optional)* | 3B | Colab A100 GPU — best quality |

How this is achieved technically:
- Models are loaded via Hugging Face `AutoModelForSeq2SeqLM` / `AutoTokenizer`.
- The same NER prompts are sent to each model.
- Outputs are parsed into structured entity predictions.
- Generation uses beam search (`num_beams=4`) for more coherent structured output.

This satisfies the "use LLM and compare model behavior under prompt changes" requirement.

---

## 5. Fast Test vs Full Run Toggle

To support fast debugging and full evaluation, the notebook includes:

- `USE_FAST_MODE = True` for quick tests
- `FAST_SENTENCE_LIMIT = 80` for small subset testing
- `FULL_SENTENCE_LIMIT = None` for full-file mode

Behavior:
- If `USE_FAST_MODE` is `True`, only first `FAST_SENTENCE_LIMIT` sentences are evaluated.
- If `USE_FAST_MODE` is `False`, full evaluation is run (or capped by `FULL_SENTENCE_LIMIT` if set).

This provides a clean switch between development and final runs.

---

## 6. Evaluation Protocol

### Output Format
The model is instructed to return only:
- JSON array of objects
- each object has keys `entity` and `type`
- `type` must be one of `PER`, `ORG`, `LOC`, `MISC`

### Metrics
1. **Entity-level Precision**
2. **Entity-level Recall**
3. **Entity-level F1**
4. **Valid JSON Rate**
5. **Exact Sentence Match Rate**

### Comparisons
- Prompt effect (0-shot vs few-shot)
- Model effect (flan-t5-small vs flan-t5-base)

### Test Run Results (Fast Mode)
Run date: 2026-03-16

Configuration used:
- `USE_FAST_MODE = True`
- `FAST_SENTENCE_LIMIT = 80`
- Evaluated sentences: 80
- Full evaluation size in `eng.testa`: 3,250 sentences

Per-prompt metrics from test run:

| Model | Prompt Template | Precision | Recall | F1 | Valid JSON Rate | Exact Sentence Match |
|---|---|---:|---:|---:|---:|---:|
| google/flan-t5-small | zero_shot_plain | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1250 |
| google/flan-t5-small | zero_shot_rubric | 0.0000 | 0.0000 | 0.0000 | 0.3125 | 0.1250 |
| google/flan-t5-small | few_shot_compact | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 0.1250 |
| google/flan-t5-small | few_shot_balanced | 0.0000 | 0.0000 | 0.0000 | 0.8250 | 0.1250 |
| google/flan-t5-base | zero_shot_plain | 0.0000 | 0.0000 | 0.0000 | 0.3250 | 0.1250 |
| google/flan-t5-base | zero_shot_rubric | 0.0000 | 0.0000 | 0.0000 | 0.0750 | 0.1250 |
| google/flan-t5-base | few_shot_compact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1250 |
| google/flan-t5-base | few_shot_balanced | 0.0000 | 0.0000 | 0.0000 | 0.1875 | 0.1250 |

0-shot vs few-shot summary:

| Model | Shot Type | Precision | Recall | F1 | Valid JSON Rate | Exact Sentence Match |
|---|---|---:|---:|---:|---:|---:|
| google/flan-t5-base | few-shot | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.1250 |
| google/flan-t5-base | zero-shot | 0.0000 | 0.0000 | 0.0000 | 0.2000 | 0.1250 |
| google/flan-t5-small | few-shot | 0.0000 | 0.0000 | 0.0000 | 0.4187 | 0.1250 |
| google/flan-t5-small | zero-shot | 0.0000 | 0.0000 | 0.0000 | 0.6563 | 0.1250 |

Interpretation:
- The pipeline execution is correct (data loading, prompting, parsing, logging, and metric computation all completed).
- Extraction quality is very low for this model/prompt setup on NER (F1 = 0 in this fast smoke run).
- Output-format reliability differs by prompt design (valid JSON rate varies strongly).

### Test Run Timing and Full-Run Estimate
Measured timings from this fast run:
- Model load time: 2.023 seconds
- Metric evaluation time (80 sentences): 342.843 seconds
- Log generation time: 33.991 seconds
- Total test run time: 378.857 seconds (6.31 minutes)

Estimated full-run time for `eng.testa` (3,250 sentences), based on observed per-call speed:
- Estimated full metric evaluation time: 13,927.993 seconds
- Estimated full total (evaluation + logs, excluding fresh model download): 13,961.984 seconds
- Estimated full total in minutes: 232.70 minutes
- Estimated full total in hours: 3.88 hours

Note:
- First full run on a new machine can be longer due to model download/cache.
- After caching, runtime should be closer to the estimate above.

---

## 7. Test Run Analysis and Prompt Improvements

### Root Cause Analysis (Initial Test Run, F1 = 0)

The initial fast run (80 sentences, greedy decoding) produced F1 = 0 for all model/prompt combinations. Inspection of `ner_prompt_output_logs.csv` revealed three distinct failure modes:

| Failure mode | Example output | Root cause |
|---|---|---|
| Always-empty array | `[]` | Model not recognising ALL-CAPS tokens as named entities |
| Hallucinated code | `a = 0 a = 0 b = 0 c = 0 for i in ...` | `zero_shot_rubric` too minimal; model reverted to code generation |
| Malformed JSON | `["ENTITY":"LEICESTERSHIRE..."]` | Model confusing `{}` object notation with `["key":"val"]` |

### Improvements Applied

| Change | Before | After |
|---|---|---|
| Prompt examples | Normal-case text ("Barack Obama") | ALL-CAPS text matching CoNLL format |
| `zero_shot_rubric` | 2-line minimal prompt | Per-type definitions + explicit "No code" rule |
| `few_shot_compact` | Renamed to `few_shot_caps` + domain-relevant examples | |
| Decoding | Greedy (`do_sample=False`) | Beam search (`num_beams=4`, `early_stopping=True`) |
| `max_new_tokens` | 128 | 256 |
| Tokenizer `max_length` | 512 | 1024 |
| Larger model support | Not available | `flan-t5-large` / `flan-t5-xl` (commented, for Colab GPU) |

---

## 8. Evidence / Logs

The notebook stores prediction logs for qualitative analysis:
- DataFrame preview in notebook (`logs_df`)
- CSV export: `ner_prompt_output_logs.csv`

This satisfies the requirement to show output differences for different prompt formulations.

---

## 9. Bias Risks and Mitigation

### Potential Biases
1. Over-representation of certain name/geography patterns in news corpora.
2. Prompt-example bias toward specific entity types.
3. Domain mismatch when moving from news text to other domains.

### Mitigation
1. Use balanced few-shot examples across entity types.
2. Include no-entity examples (`[]`) to reduce over-tagging.
3. Evaluate slices by entity type and domain.
4. Keep strict output schemas with parser checks.

---

## 10. Conclusion

This assignment demonstrates prompt engineering for NER with a local CoNLL-2003 dataset, relative-path loading, model comparison, and fast/full evaluation toggle.

Current status after fast test:
- Functional status: pass
- Quality status on current setup: weak (needs model/prompt improvements before final-quality submission)

Deliverables covered:
1. Jupyter notebook with multiple prompt designs: complete
2. Comparative analysis framework: complete
3. Logs for different outputs across prompt formulations: complete
4. Bias discussion and mitigation: complete
5. Multi-model comparison under same prompt setting: complete
