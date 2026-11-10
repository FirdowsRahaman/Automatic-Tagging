# Automatic Multi-Label Document Tagging

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![NLP: Multi-Label](https://img.shields.io/badge/NLP-Multi--Label%20Classification-indigo.svg)](#)

Automated multi-label document tag prediction and topic categorization pipeline using Natural Language Processing.

## Overview
An end-to-end NLP framework designed to predict relevant topic tags for technical text documents and developer questions (e.g., StackOverflow-style multi-tagging). Employs TF-IDF / embedding representations and multi-label classifiers (Binary Relevance, Classifier Chains) with micro/macro F1 evaluation.

## Key Capabilities
- **Text Normalization:** HTML stripping, code-block tokenization, stopword filtering, and lemmatization.
- **Multi-Label Modeling:** OneVsRest classification with calibrated probabilities for multi-label assignment.
- **Evaluation:** Precision@k, Recall@k, Hamming Loss, and Jaccard similarity metrics.

## Architecture
- `trainer/`: Modular model training scripts, preprocessing pipelines, and evaluation routines.

## Quickstart
```bash
git clone https://github.com/FirdowsRahaman/Automatic-Tagging.git
cd Automatic-Tagging
pip install scikit-learn pandas numpy nltk
python trainer/train.py
```

## License
This project is licensed under the MIT License.
