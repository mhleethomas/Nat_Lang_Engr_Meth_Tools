Assignment 2: Named Entity Recognition (NER)

Dataset
- Dataset: CoNLL-2003 (English)
- Split sizes: eng.train / eng.testa (validation) / eng.testb (test)

Models
- Model 1: Conditional Random Field (CRF) with handcrafted lexical and POS features
- Model 2: BiLSTM tagger with word embeddings and a linear classifier

Hyperparameter Tuning
- CRF: grid over C1 (L1) and C2 (L2)
- BiLSTM: small sweep over hidden size, dropout, and learning rate

Evaluation Metrics
- Entity-level Precision, Recall, F1 (seqeval)
- Per-entity breakdown (ORG, PER, LOC, MISC)

Results
- CRF (best on validation):
  - Validation F1: 0.8836
  - Test Precision/Recall/F1: 0.7977 / 0.7835 / 0.7905
- BiLSTM (best on validation):
  - Validation F1: 0.6903
  - Test Precision/Recall/F1: 0.6954 / 0.5837 / 0.6347

Error Analysis
- Common error patterns:
  - ORG vs LOC confusions on country and organization names
  - ORG vs PER confusions on person names in organization contexts
  - Boundary errors on multi-token entities

Findings
- CRF performs strongly with rich local features and POS signals.
- BiLSTM captures longer-range context but needs more tuning and epochs.

Limitations
- Limited hyperparameter search due to runtime constraints.
- No character-level features; may reduce robustness to rare or misspelled tokens.

Future Work
- Add character CNN or BiLSTM-CRF for better boundary modeling.
- Use pretrained embeddings (e.g., GloVe) and/or contextual encoders.
