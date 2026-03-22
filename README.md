# Multilingual Sentiment Analysis using LLMs — IIT Guwahati

Research internship under Dr. S. Ranbir Singh, IIT Guwahati.

The goal was simple: how well do multilingual LLMs actually handle Indian languages, especially low-resource ones? We ran sentiment classification experiments across 6 Indian languages using multiple models, with very limited training data to simulate real-world low-resource conditions.

---

## Languages Covered

Hindi, Bengali, Tamil, Telugu, Urdu, Assamese

---

## Models Evaluated

| Model | Source |
|---|---|
| mBERT (bert-base-multilingual-uncased) | Google |
| MuRIL (google/muril-base-cased) | Google |
| Navarasa | Community (Indian language focused) |
| IndicBERT v2 | AI4Bharat |
| PaLM-2 | Google |
| XLM-R | Meta |

---

## What We Tested

**Training data splits** — to simulate low-resource conditions:
- 1% of data (~52 samples)
- 2% of data (~104 samples)
- 10% of data (~520 samples)

**Techniques applied:**
- Prompt tuning (via PEFT library) with 8 virtual tokens
- Few-shot learning
- Standard fine-tuning for comparison

**Task:** Binary sentiment classification (positive / negative)

---

## Notebook Structure

```
mBERT.ipynb                          # mBERT baseline setup
mBERT_final.ipynb                    # mBERT final experiment
mBert-Sample-accuracy-bn.ipynb       # mBERT on Bengali
mBERT-Sample-accuracy-hi.ipynb       # mBERT on Hindi
mBertSample1.ipynb                   # mBERT sample run 1
mBERTsample2.ipynb                   # mBERT sample run 2
mBert_promptTuning.ipynb             # mBERT with prompt tuning

MuRIL-Sample--Split.ipynb            # MuRIL with train/test split
MuRIL-Sample-accuracy-hi.ipynb       # MuRIL on Hindi
MuRIL-Sample-accuracy-bn.ipynb       # MuRIL on Bengali
MuRIL-Sample-accuracy-en.ipynb       # MuRIL on English (baseline)
MuRIL-Sample-accuracy-Hindi2.ipynb   # MuRIL Hindi iteration 2
MuRIL-Sample-accuracy-ta             # MuRIL on Tamil
MuRIL_Sample_accuracy_Hindi_Improvement1.ipynb  # Hindi accuracy improvement run

navarasa.ipynb                       # Navarasa model experiments
PaLM-2-Sample.ipynb                  # PaLM-2 prompt tuning experiment
```

---

## Key Result

IndicBERT v2 achieved the best overall accuracy at **0.766**, outperforming larger general-purpose multilingual models including mBERT and MuRIL on Indian language sentiment tasks.

---

## Setup

```bash
pip install peft transformers datasets torch scikit-learn pandas tqdm
pip install pyarrow==8.0.0
```

All experiments were run on Google Colab with a T4 GPU.

---

## Dataset

Experiments used the `societal_2l` dataset — a multilingual sentiment dataset with language-specific CSV files per language (e.g. `societal_2l_train_hi.csv` for Hindi). Labels are binary: `0` (negative) and `4` / `1` (positive, remapped to 1).

Dataset files were stored on Google Drive and mounted in Colab.

---

## Training Config

```python
num_virtual_tokens = 8
prompt_tuning_init_text = "What is the sentiment of this text?"
max_length = 128
lr = 3e-2
batch_size = 8
num_epochs = 20
train/test split = 80/20
```

---

## Publication

> *How multilingual LLMs are! A case study of LLMs using multilingual Sentiment Analysis in Indian Language*
> RIC 2024 — under Dr. S. Ranbir Singh, IIT Guwahati

---

## Author

**Anubhav Bhattacharya**
M.Tech CSE, Rajiv Gandhi Institute of Petroleum Technology
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)
