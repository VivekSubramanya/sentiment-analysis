# Sentiment & Topic Analysis on Twitter Conversations

## Overview
This repository implements a **Python NLP pipeline** for analyzing customer sentiment in Twitter conversations. It replicates, in a fully open‑source form, work originally done in a corporate setting — but without any pay‑to‑use LLMs. The stack is built on **spaCy** and **BERT**, with **BERTopic** for topic discovery.

The pipeline processes raw tweets, groups them into conversations, and outputs:

- Cleaned text stored in SQLite
- Topics discovered via BERTopic
- Sentiment classification (negative, neutral, positive)
- Confidence metrics and priority flags

The dataset is from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter/data, which contains a total of 2.8 million tweets. 

## Workflow

### **Preprocessing**

- Removes URLs, mentions, hashtags, emojis, and noise
- Normalizes text and stores results in SQLite

### **Analysis**

- Aggregates tweets into conversations
- Generates embeddings with SentenceTransformers
- Applies BERTopic for topic modeling
- Runs transformer‑based sentiment classification
- Assigns priority labels based on confidence thresholds

## Packages Used

- **pandas**, **numpy**
- **spaCy** (with `en_core_web_sm`)
- **sentence-transformers**
- **BERTopic**
- **torch**, **transformers**
- **scikit-learn**
- **sqlite3** (standard library)

## Installation

### 1. Create and activate a virtual environment
Windows:

bash

```
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

bash

```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
bash

```
pip install -r requirements.txt
```

### 3. Install spaCy model
bash

```
python -m spacy download en_core_web_sm
```

> If using GPU, ensure PyTorch matches your CUDA setup.

### 4. Run the runner script
bash

```
python runner.py
```

## Confidence Metrics
The pipeline outputs `confidence_metrics.txt` with:

- Total conversations processed
- Average & median topic confidence
- Average & median sentiment confidence
- Topic distribution summary
- Sentiment distribution summary
- High vs. normal priority counts

**Example run (798k conversations from 2.8m tweets):**

- Avg topic confidence: 0.638
- Median topic confidence: 0.790
- Avg sentiment confidence: 0.684
- Median sentiment confidence: 0.662
- High priority cases: 305,432
- Normal priority cases: 492,580

Sentiment distribution:

- Negative: 350,195
- Neutral: 261,425
- Positive: 186,392

## Performance

- Runtime: ~3.5 hours on RTX 4080 + Ryzen 7 5800X + 16 GB DDR5 RAM
- Median confidence values remain stable across runs (topics ~78–82%, sentiment ~66–68%)

## Next Steps

- Tune BERTopic parameters for cleaner topics
- Experiment with alternative embedding models
- Add CLI / notebook workflow
- Expand reporting & visualization
- Integrate into larger NLP pipelines
