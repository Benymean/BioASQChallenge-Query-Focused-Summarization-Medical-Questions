# BioASQChallenge-Query-Focused-Summarization-Medical-Questions

This repository contains a set of tools for analyzing and summarizing medical questions and their corresponding answers. It is designed to facilitate research and development in the field of biomedical natural language processing.

## Project Components

### 1. Jupyter Notebook: Analysis of Medical Questions
The Jupyter notebook (`Query-Focused Summarization Project on Medical Questions.ipynb`) provides a platform for initial data exploration and analysis, primarily using the Pandas library. It includes operations for reading and manipulating data, specifically targeting the "bioasq10b_labelled.csv" dataset.

### 2. Main Script: Text Analysis (`a3_1.py`)
This script offers various functions for detailed text analysis:
- **Part-of-Speech Statistics**: Analyze the distribution of parts of speech in medical questions.
- **Top Stemmed N-grams**: Identify and rank the top stemmed n-grams in the dataset.
- **N-gram Calculation**: General function to calculate n-grams.
- **Named Entity Recognition**: Extract named entities from the text.
- **TF-IDF Analysis**: Compute TF-IDF scores for text comparison.

### 3. Test Suite (`test_a3_1.py`)
A suite of unit tests to validate the functionalities implemented in `a3_1.py`:
- Tests cover various aspects including n-gram calculations, named entity recognition, and TF-IDF analysis.

## Setup and Installation
Ensure you have Python 3.x installed, then install the required libraries:
```bash
pip install pandas numpy nltk spacy sklearn

