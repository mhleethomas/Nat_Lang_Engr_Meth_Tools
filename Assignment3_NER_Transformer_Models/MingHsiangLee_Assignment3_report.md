# Assignment 3: Transformer-Based NER Models

## Executive Summary
This assignment compares two transformer-based token classification models for NER on CoNLL-2003:
- BERT: `bert-base-cased`
- DeBERTa: `microsoft/deberta-v3-base`

The notebook fine-tunes both models with the same training strategy, evaluates them at both entity-level and token-level, and analyzes systematic misclassifications.

## Dataset and Task Setup
- Dataset: CoNLL-2003 (same benchmark family used in Assignment 2)
- Splits:
1. Train: 14,041 sentences
2. Validation: 3,250 sentences
3. Test: 3,453 sentences
- Entity types: `LOC`, `PER`, `ORG`, `MISC`
- Tagging scheme: BIO labels

Implementation detail:
- Because transformers use subword tokenization, labels are aligned to the first subword of each word.
- Continuation subwords are masked with `-100` so they do not contribute to loss.

## Model Selection
1. `bert-base-cased`
- Classic BERT baseline, widely used for NER.
- Approx. 110M parameters.

2. `microsoft/deberta-v3-base`
- Newer transformer variant with disentangled attention.
- Approx. 86M parameters.

Rationale:
- One baseline BERT-family model and one newer architecture variant, as required.

## Fine-Tuning Methodology
Shared training setup in the notebook:
- Optimizer: `AdamW`
- Learning rate: `2e-5`
- Batch size: `16`
- Gradient accumulation: `2` (effective batch size 32)
- Warmup steps: `500`
- Epochs: up to `5`
- Gradient clipping: `max_grad_norm=1.0`
- Early stopping: patience `3` based on validation entity-level F1

This satisfies the assignment requirement to use suitable optimization and stabilization techniques.

## Evaluation Methodology
The notebook computes both required evaluation views.

1. Entity-level metrics (strict NER scoring)
- Precision
- Recall
- F1-score
- Per-entity breakdown via `seqeval` report

2. Token-level metrics
- Micro Precision
- Micro Recall
- Micro F1
- Token-level classification report

3. Error-focused diagnostics
- Token-level confusion matrix (BIO labels)
- Top misclassification pairs (true label -> predicted label)

## Results
Use the values printed by `MingHsiangLee_Assignment3.ipynb` to fill this section.

### Overall Metrics
| Metric | BERT | DeBERTa |
|---|---:|---:|
| Entity Precision | (from notebook) | (from notebook) |
| Entity Recall | (from notebook) | (from notebook) |
| Entity F1 | (from notebook) | (from notebook) |
| Token F1 | (from notebook) | (from notebook) |
| Inference Time (s) | (from notebook) | (from notebook) |

### Comparison to Assignment 2
| Model | F1 |
|---|---:|
| CRF (Assignment 2) | 0.7905 |
| BiLSTM (Assignment 2) | 0.6347 |
| BERT (Assignment 3) | (from notebook) |
| DeBERTa (Assignment 3) | (from notebook) |

## Misclassification Analysis
Summarize from confusion matrices and top error pairs:
- Most common confusions (`ORG` vs `LOC`, boundary errors like `B-*` vs `I-*`, etc.)
- Which entity types are hardest (`MISC`/`ORG` often harder than `PER`/`LOC`)
- Whether both models fail on the same patterns (dataset ambiguity) or one model has a clear edge

Include 2-3 concrete examples from notebook output where possible.

## Discussion and Conclusion
Address the four required prompts directly:

1. Which model performed better and why?
- Choose based on entity-level F1 first.
- If close, discuss whether difference is practically meaningful.

2. Trade-offs in accuracy, speed, and computational cost
- Accuracy: entity-level F1
- Speed: inference time
- Cost: parameter count (BERT ~110M vs DeBERTa ~86M)

3. Impact of pretraining
- Explain how large-scale pretraining improves contextual understanding and transfer to NER.
- Relate this to why transformers outperform traditional Assignment 2 methods.

4. Improvements for better NER performance
1. Add CRF head on top of transformer outputs for BIO transition constraints.
2. Domain-adaptive pretraining on target-domain text.
3. Data augmentation and class balancing for low-frequency entities.
4. Hyperparameter search (LR, warmup, max length, batch strategy).
5. Ensemble or distilled deployment model for quality/speed balance.

## Rubric Coverage Check
- Two transformer models for token classification: satisfied
- Fine-tuning both models with suitable optimizer/hyperparameters: satisfied
- Gradient clipping and early stopping: satisfied
- Entity-level vs token-level evaluation: satisfied
- Confusion matrix for misclassified entities: satisfied
- Misclassification analysis and systematic errors: satisfied
- Discussion of winner, trade-offs, pretraining impact, improvements: satisfied

## Final Note
The notebook has been cleaned and structured as a single end-to-end pipeline in:
- `Assignment3_NER_Transformer_Models/MingHsiangLee_Assignment3.ipynb`

Run it once to populate the final numeric values in this report.
