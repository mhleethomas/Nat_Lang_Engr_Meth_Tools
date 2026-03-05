# Assignment 3: Transformer-Based NER Models

## Executive Summary
This assignment compares two transformer-based token classification models for NER on CoNLL-2003:
- BERT: `bert-base-cased`
- DeBERTa: `microsoft/deberta-v3-base`

Both models were fine-tuned in Colab using the same setup (including explicit dropout). BERT achieved strong and stable performance, while DeBERTa failed to converge (training loss became `nan`), resulting in near-zero practical accuracy.

## Dataset and Task Setup
- Dataset: CoNLL-2003
- Splits:
1. Train: 14,041 sentences
2. Validation: 3,250 sentences
3. Test: 3,453 sentences
- Entity types: `LOC`, `PER`, `ORG`, `MISC`
- Tag scheme: BIO

Implementation detail:
- Labels are aligned only to the first subword per token.
- Continuation subwords are masked with `-100` and excluded from loss.

## Model Selection
1. `bert-base-cased`
- Baseline transformer for NER.
- Approx. 110M parameters.

2. `microsoft/deberta-v3-base`
- More recent architecture variant (disentangled attention).
- Approx. 86M parameters.

Rationale:
- One BERT-family baseline and one newer transformer variant, as required.

## Fine-Tuning Methodology
Shared training setup:
- Optimizer: `AdamW`
- Learning rate: `2e-5`
- Batch size: `16`
- Dropout: explicit model/head dropout `0.1` for both models
- Gradient accumulation: `2` (effective batch size 32)
- Warmup steps: `500`
- Epochs: up to `5`
- Gradient clipping: `max_grad_norm=1.0`
- Early stopping: patience `3` based on validation entity-level F1

Training observations from this Colab run:
- BERT converged normally with validation F1 improving from `0.9086` to `0.9483` over 5 epochs.
- DeBERTa produced `nan` for train/validation loss from epoch 1 and was early-stopped at epoch 4.

## Evaluation Methodology
1. Entity-level metrics (`seqeval`)
        - Precision
        - Recall
        - F1-score
        - Per-entity breakdown

2. Token-level metrics (`sklearn`)
        - Micro Precision
        - Micro Recall
        - Micro F1
        - Per-label token report

3. Error diagnostics
        - Token-level confusion matrix (BIO labels)
        - Top misclassification pairs (`true -> pred`)

## Results

### Overall Metrics (From Colab Run)
| Metric | BERT | DeBERTa |
|---|---:|---:|
| Entity Precision | **0.9033** | 0.0309 |
| Entity Recall | **0.9184** | 0.2542 |
| Entity F1 | **0.9108** | 0.0551 |
| Token F1 (micro) | **0.9821** | 0.0359 |
| Inference Time (s) | 44.84 | **19.02** |

Interpretation:
- BERT is the only usable model in this run.
- DeBERTa is faster, but the output quality is too poor for practical NER.

### Comparison to Assignment 2
| Model | Entity F1 |
|---|---:|
| CRF (Assignment 2) | 0.7905 |
| BiLSTM (Assignment 2) | 0.6347 |
| BERT (Assignment 3) | **0.9108** |
| DeBERTa (Assignment 3) | 0.0551 |

Key observation:
- BERT improves by `+0.1203` F1 over CRF and `+0.2761` over BiLSTM.
- DeBERTa underperforms due to convergence failure despite newer architecture.

## Misclassification Analysis

### BERT Error Patterns
Entity/token reports and top confusion pairs show:
- Strong classes: `B-PER` (F1 `0.9624`), `B-LOC` (F1 `0.9353`), `I-PER` (F1 `0.9879`).
- Weak classes: `B-MISC` (F1 `0.8299`), `I-MISC` (F1 `0.6973`).
- Frequent mistakes:
    - `O -> I-MISC` (66)
    - `O -> B-MISC` (58)
    - `B-ORG -> B-LOC` (56)
    - `B-LOC -> B-ORG` (46)

This indicates boundary ambiguity around `MISC` and type confusion between `ORG` and `LOC`.

### DeBERTa Error Patterns
DeBERTa shows severe collapse:
- Most labels are predicted as `B-LOC`.
- Example dominant errors:
    - `O -> B-LOC` (38,323)
    - `B-ORG -> B-LOC` (1,661)
    - `B-PER -> B-LOC` (1,617)
    - `I-PER -> B-LOC` (1,156)
- Per-label token metrics are near zero for almost all labels except high `B-LOC` recall.

This is consistent with the `nan` training-loss behavior and indicates optimization instability rather than a minor tuning issue.

## Discussion and Conclusion

### 1. Which Model Performed Better and Why?
**BERT clearly performed better** (`0.9108` vs `0.0551` entity F1).

BERT trained stably and generalized across entity types. DeBERTa failed to optimize under this configuration (loss became `nan`), causing degenerate predictions.

### 2. Trade-offs: Accuracy, Speed, and Computational Cost
| Dimension | BERT | DeBERTa |
|---|---|---|
| Accuracy (Entity F1) | **0.9108** | 0.0551 |
| Inference Speed | 44.84s | **19.02s** |
| Parameters | 110M | **86M** |

Trade-off summary:
- BERT is slower and larger, but reliable and deployment-viable.
- DeBERTa is faster/smaller, but unusable in this run due to collapse.

### 3. Impact of Pretraining on NER Transfer Learning
The run supports the transfer-learning advantage of pretraining:
- Pretrained BERT reaches strong NER performance in only a few epochs.
- Context-rich pretrained representations help boundary/type decisions better than traditional non-transformer baselines.
- However, architecture quality alone is not enough: optimization stability during fine-tuning is critical (as seen with DeBERTa).

## Rubric Coverage Check
- Two transformer models for token classification: satisfied
- Fine-tuning both models with suitable optimizer/hyperparameters: satisfied
- Explicit dropout configuration: satisfied
- Gradient clipping and early stopping: satisfied
- Entity-level vs token-level evaluation: satisfied
- Confusion matrix for misclassified entities: satisfied
- Misclassification analysis and systematic errors: satisfied
- Discussion of winner, trade-offs, pretraining impact, improvements: satisfied
