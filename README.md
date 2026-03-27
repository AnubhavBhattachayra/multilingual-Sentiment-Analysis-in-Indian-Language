# How Multilingual LLMs Are! A Case Study of LLMs Using Multilingual Sentiment Analysis in Indian Language 🇮🇳

[![Paper Link](https://img.shields.io/badge/Paper-Springer%20Link-blue)](https://link.springer.com/chapter/10.1007/978-981-96-8753-4_34)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--981--96--8753--4__34-green)](https://doi.org/10.1007/978-981-96-8753-4_34)
[![Conference](https://img.shields.io/badge/Conference-RIC%202024-orange)](#)

*Research internship under Dr. S. Ranbir Singh, IIT Guwahati.*

This repository contains the code and notebooks for our research paper: **"How Multilingual LLMs Are! A Case Study of LLMs Using Multilingual Sentiment Analysis in Indian Language"**, published in *Current Progress in Engineering Sciences (RIC 2024)* by Springer.

---

## 🎯 Abstract & Research Goal

The primary goal was to investigate: **how well do multilingual LLMs actually handle Indian languages, especially low-resource ones?** 

Large Language Models (LLMs) achieve general-purpose language understanding by training billions of parameters. However, there is a prominent lack of Multilingual LLMs with dedicated support for Indian dialects. In this study, we assessed the performance of LLMs across 6 Indian languages, simulating real-world low-resource conditions by utilizing very limited training data. 

We hand-picked multilingual LLMs (including XLM-R, mBERT, MuRIL, Navarasa, IndicBERT v2, and mT5) and fine-tuned them using prompt tuning and few-shot learning for Sentiment Analysis. By determining whether the emotional tone of text is positive or negative, this task provided a robust test of the models' deep natural language understanding capabilities.

## 📄 Read the Paper

- **Full Text**: [Available on Springer Link](https://link.springer.com/chapter/10.1007/978-981-96-8753-4_34)
- **Alternate Link**: [Google Scholar](https://scholar.google.com/scholar?oi=bibs&cluster=7153842800473556670&btnI=1&hl=en)
- **DOI**: [`10.1007/978-981-96-8753-4_34`](https://doi.org/10.1007/978-981-96-8753-4_34)

### Citation
If you find this repository or our research useful, please cite our paper:
```bibtex
@inproceedings{kumar2026how,
  title={How Multilingual LLMs Are! A Case Study of LLMs Using Multilingual Sentiment Analysis in Indian Language},
  author={Kumar, S. and Agrahari, S. and Singh, S. R. and Saikia, A. and Choudhury, R. and Bhattacharya, A.},
  booktitle={Current Progress in Engineering Sciences. RIC 2024},
  editor={Dixit, U. S. and Bharadwaj, N. and Kumar, S. and Sikdar, D.},
  publisher={Springer, Singapore},
  year={2026},
  doi={10.1007/978-981-96-8753-4_34},
  url={https://doi.org/10.1007/978-981-96-8753-4_34}
}
```

---

## 🌐 Languages Covered

Hindi (हिन्दी) | Bengali (বাংলা) | Tamil (தமிழ்) | Telugu (తెలుగు) | Urdu (اردو) | Assamese (অসমীয়া)

---

## 🤖 Models Evaluated

| Model | Source | Description |
|---|---|---|
| **mBERT** (`bert-base-multilingual-uncased`) | Google | Standard multilingual baseline |
| **MuRIL** (`google/muril-base-cased`) | Google | Specifically trained on Indian languages |
| **Navarasa** | Community | Indian language focused multilingual model |
| **IndicBERT v2** | AI4Bharat | State-of-the-art model for Indic languages |
| **PaLM-2** | Google | Advanced commercial LLM |
| **XLM-R** | Meta | Broad coverage multilingual model |
| **mT5** | Google | Massively multilingual pre-trained transformer *(Evaluated in the full study)*|

---

## 🔬 Experimental Setup & Techniques

**Task:** Binary sentiment classification (positive / negative).

**Training Data Splits:** Simulating severe low-resource conditions:
- **1%** of data (~52 samples)
- **2%** of data (~104 samples)
- **10%** of data (~520 samples)

**Techniques Applied:**
- **Prompt tuning** (via PEFT library) using 8 virtual tokens. 
  - *Prompt*: `"What is the sentiment of this text?"`
- **Few-shot learning**
- Standard fine-tuning (for baseline comparisons)

---

## 🏆 Key Result

**IndicBERT v2** achieved the best overall accuracy at **0.766**, outperforming larger, general-purpose multilingual models including mBERT and MuRIL on Indian language sentiment tasks. This successfully demonstrates the superiority of models with language-specific continued pre-training for regional dialects.

---

## 📂 Repository / Notebook Structure

```text
├── Baselines/
│   ├── mBERT.ipynb                          # mBERT baseline setup
│   ├── mBERT_final.ipynb                    # mBERT final experiment
│   ├── mBert-Sample-accuracy-bn.ipynb       # mBERT on Bengali
│   ├── mBERT-Sample-accuracy-hi.ipynb       # mBERT on Hindi
│   ├── mBertSample1.ipynb                   # mBERT sample run 1
│   ├── mBERTsample2.ipynb                   # mBERT sample run 2
│   └── mBert_promptTuning.ipynb             # mBERT with prompt tuning
├── MuRIL/
│   ├── MuRIL-Sample--Split.ipynb            # MuRIL with train/test split
│   ├── MuRIL-Sample-accuracy-hi.ipynb       # MuRIL on Hindi
│   ├── MuRIL-Sample-accuracy-bn.ipynb       # MuRIL on Bengali
│   ├── MuRIL-Sample-accuracy-en.ipynb       # MuRIL on English (baseline)
│   ├── MuRIL-Sample-accuracy-Hindi2.ipynb   # MuRIL Hindi iteration 2
│   ├── MuRIL-Sample-accuracy-ta.ipynb       # MuRIL on Tamil
│   └── MuRIL_Sample_accuracy_Hindi_Improvement1.ipynb  # Hindi accuracy improvement run
├── Other_Models/
│   ├── navarasa.ipynb                       # Navarasa model experiments
│   └── PaLM-2-Sample.ipynb                  # PaLM-2 prompt tuning experiment
```

---

## ⚙️ Setup & Installation

All experiments were executed on **Google Colab** using an NVIDIA T4 GPU.

### Requirements Installation
```bash
pip install peft transformers datasets torch scikit-learn pandas tqdm
pip install pyarrow==8.0.0
```

### Dataset Structure
Experiments utilized the `societal_2l` dataset — a multilingual sentiment dataset comprising language-specific CSV files per language (e.g., `societal_2l_train_hi.csv` for Hindi). 
- Labels are binary: `0` (negative) and `4` / `1` (positive, remapped to `1`).
- Used via Google Drive mounting in Colab.

### Training Configuration

```python
num_virtual_tokens = 8
prompt_tuning_init_text = "What is the sentiment of this text?"
max_length = 128
lr = 3e-2
batch_size = 8
num_epochs = 20
train_test_split = "80/20"
```

---

## ✍️ Authors

- **Authors**: S. Kumar, S. Agrahari, S. R. Singh (IIT Guwahati)
- **Co-authors**: A. Saikia, R. Choudhury, **Anubhav Bhattacharya**

*Note from Anubhav Bhattacharya: I served as a co-author on this paper during my research internship under Dr. S. Ranbir Singh at IIT Guwahati. I am currently pursuing an M.Tech in CSE at Rajiv Gandhi Institute of Petroleum Technology.*
- [LinkedIn](#) *(Add your LinkedIn URL here)*
- [GitHub](#) *(Add your GitHub URL here)*
