# Nat_Lang_Engr_Meth_Tools

Course assignments for natural language engineering methods and tools.

## Structure

- **Assignment1**: Sentiment analysis with Sentiment140 dataset
- **Assignment2_Name_Entity_Recognition**: Named Entity Recognition using WikiAnn and CoNLL-2003
- **Assignment3_NER_Transformer_Models**: Transformer-based NER with BERT and DeBERTa

## Setup

### Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### Install Dependencies

#### Assignment 1: Sentiment Analysis
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

#### Assignment 2: Named Entity Recognition
```bash
pip install pandas numpy scikit-learn seqeval sklearn-crfsuite datasets
```

#### Assignment 3: Transformer-Based NER
```bash
pip install transformers torch seqeval scikit-learn matplotlib seaborn pandas numpy tqdm sentencepiece protobuf
```

**Note**: Assignment 3 requires significant computational resources (GPU recommended). For quick testing on CPU, the notebook includes an MVP mode with reduced dataset size and epochs.

## Usage

1. Activate the virtual environment
2. Open the notebook for the assignment in Jupyter or VS Code
3. Run cells sequentially from top to bottom

## Assignment Details

### Assignment 1: Sentiment Analysis
- Dataset: Sentiment140 (1.6M tweets)
- Models: Logistic Regression, Naive Bayes, SVM
- Features: TF-IDF, n-grams

### Assignment 2: Named Entity Recognition
- Datasets: WikiAnn-en, CoNLL-2003
- Models: CRF (Conditional Random Fields)
- Evaluation: Entity-level precision, recall, F1

### Assignment 3: Transformer-Based NER
- Dataset: CoNLL-2003
- Models: BERT (bert-base-cased), DeBERTa (microsoft/deberta-v3-base)
- Training: Fine-tuning with AdamW, gradient clipping, early stopping
- Evaluation: Entity-level & token-level metrics, confusion matrices

**MVP Mode**: For quick validation on CPU, set `MVP_TRAIN_LIMIT = 2000` and `num_epochs = 2` in the notebook (already configured by default).

**Full Training**: For production results, set `MVP_TRAIN_LIMIT = None` and `num_epochs = 5`. Recommended to run overnight with GPU.

## Notes

- Large CSV files are excluded from git (see `.gitignore`)
- Model checkpoints (`.pt`, `.pth`) are not tracked
- HuggingFace model cache is stored in `~/.cache/huggingface/`
