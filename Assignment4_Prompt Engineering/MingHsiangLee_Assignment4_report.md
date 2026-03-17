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

### Latest Local CPU Run Results
Run date: 2026-03-17

Configuration used:
- Local CPU
- Models: `google/flan-t5-small`, `google/flan-t5-base`
- Prompts: `zero_shot_plain`, `zero_shot_rubric`, `few_shot_caps`, `few_shot_balanced`
- Decoding: beam search (`num_beams=4`), `max_new_tokens=256`

0-shot vs few-shot summary from the latest notebook run:

| Model | Shot Type | Precision | Recall | F1 | Valid JSON Rate | Exact Sentence Match |
|---|---|---:|---:|---:|---:|---:|
| google/flan-t5-base | few-shot | 0.0000 | 0.0000 | 0.0000 | 0.00355 | 0.1985 |
| google/flan-t5-base | zero-shot | 0.0000 | 0.0000 | 0.0000 | 0.03105 | 0.1985 |
| google/flan-t5-small | few-shot | 0.0000 | 0.0000 | 0.0000 | 0.03740 | 0.1985 |
| google/flan-t5-small | zero-shot | 0.0000 | 0.0000 | 0.0000 | 0.29370 | 0.1985 |

Interpretation and evaluation:
- **Pipeline correctness: pass.** Data loading, generation, parsing, aggregation, and logging all run successfully.
- **Entity extraction quality: fail for this setup.** Precision/Recall/F1 are 0 for all model-shot combinations.
- **Schema reliability: weak.** Even the best JSON validity (`flan-t5-small` zero-shot, 0.29370) means over 70% outputs are not parseable as valid target JSON.
- **Exact sentence match (0.1985) is not a quality win.** This is mostly consistent with frequent empty predictions (`[]`) and many no-entity or mismatch cases, not correct entity extraction.

Overall judgment for current local CPU setting:
- This is a valid and well-executed prompt-engineering experiment.
- However, prompt changes alone with these two small FLAN-T5 models are insufficient to achieve useful NER extraction quality under a strict JSON-output evaluator.

---

## 7. Error Analysis of Final Run

Qualitative inspection of log samples confirms that failures are now mainly output-format and extraction-behavior issues, not execution bugs.

Typical model outputs observed:

| Failure mode | Example output from logs | Effect |
|---|---|---|
| Label-only output | `PER`, `LOC` | Not valid JSON object list; parser returns empty prediction |
| Near-JSON but invalid syntax | `["entity": "LONDON", "type": "LOC"]` | Missing object braces; parse fails |
| Template echo / placeholder output | `["entity": "...", "type": "..."]` | Structurally wrong and semantically empty |
| Over-compressed pseudo list | `[Phil Simmons, batsman, batsman]` | Not valid schema; parse fails |

This explains why the valid JSON rate is very low and why entity-level F1 remains zero.

Key conclusion from error analysis:
- For strict NER JSON extraction, small/base FLAN-T5 on CPU is strongly constrained by generation format compliance.
- Prompt engineering improved instruction quality, but did not overcome model capacity limits for this structured extraction target.

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

Current status after latest local CPU run:
- Functional status: pass
- Quality status on current setup: weak (F1 remains 0; output-format compliance is the main bottleneck)

Deliverables covered:
1. Jupyter notebook with multiple prompt designs: complete
2. Comparative analysis framework: complete
3. Logs for different outputs across prompt formulations: complete
4. Bias discussion and mitigation: complete
5. Multi-model comparison under same prompt setting: complete
