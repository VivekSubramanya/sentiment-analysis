import pandas as pd
import sqlite3
from datetime import datetime
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from transformers import AutoTokenizer, pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


# Utility: log stage execution into performance_log
def log_stage(conn, stage, start_time, end_time, notes=""):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS performance_log (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT,
            start_time TEXT,
            end_time TEXT,
            notes TEXT
        )
    """)
    conn.execute("INSERT INTO performance_log (stage, start_time, end_time, notes) VALUES (?, ?, ?, ?)",
                 (stage, str(start_time), str(end_time), notes))
    conn.commit()

print('Start of code: ', datetime.now())
conn = sqlite3.connect('twcs_processed.db')

# 1. Load and aggregate conversations
start = datetime.now()
data = pd.read_sql('select * from tweets', conn)
sorted_data = data.sort_values(by=['conversation_id','created_at'], ascending=[True,True])
grouped_data = sorted_data.groupby('conversation_id').agg(
    Full_Conversation=('text',' || '.join),
    cleaned_tokens=('tokens',' '.join)
).reset_index()
end = datetime.now()
log_stage(conn, "Conversation Aggregation", start, end)

# 2. Prepare texts
texts_df = grouped_data[['conversation_id','Full_Conversation','cleaned_tokens']]
texts = texts_df['Full_Conversation'].dropna().astype(str).tolist()
cleaned_tokens = texts_df['cleaned_tokens'].dropna().astype(str).tolist()
cleaned_tokens = [" ".join(dict.fromkeys(s.split())) for s in cleaned_tokens]
texts_df['cleaned_tokens'] = cleaned_tokens


vectorizer_model = CountVectorizer(
    tokenizer=lambda x: x.split(),  # tokens are space-separated
    lowercase=False,                # already lowercased
    stop_words=None                 # already removed
)


# 3. Sentence-BERT embeddings on GPU (float32) with richer model
start = datetime.now()
embedder = SentenceTransformer("all-mpnet-base-v2")  # richer embeddings
use_cuda = torch.cuda.is_available()
device_embed = "cuda" if use_cuda else "cpu"
if not use_cuda:
    print("CUDA not available — using CPU for embeddings")
try:
    # Prefer numpy output to work with scikit-learn
    embeddings = embedder.encode(
        texts,
        device=device_embed,
        convert_to_numpy=True,
        show_progress_bar=True
    )
except TypeError:
    # Older sentence-transformers versions may not support `convert_to_numpy`/device
    embeddings = embedder.encode(texts, show_progress_bar=True)

# Ensure embeddings is a NumPy array
embeddings = np.asarray(embeddings)
end = datetime.now()
log_stage(conn, "Embeddings", start, end)

# 4. BERTopic with tuned parameters + cleaned tokens vectorizer
start = datetime.now()

# Ensure cleaned tokens column exists (space‑joined tokens from preprocessing)
# Example: texts_df['cleaned_tokens'] = texts_df['Full_Conversation'].apply(lambda t: " ".join(custom_cleaning(t)))
# In your case, this should already be present from Step 2 preprocessing

subset_size = 50000 if len(texts) > 50000 else len(texts)
subset_texts = texts[:subset_size]                # raw conversations
subset_embeddings = embeddings[:subset_size]

# Dimensionality reduction: apply SVD before BERTopic
svd_model = TruncatedSVD(n_components=200)
subset_embeddings_reduced = svd_model.fit_transform(subset_embeddings)
full_embeddings_reduced = svd_model.transform(embeddings)

# Custom vectorizer uses cleaned tokens instead of raw text
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(
    tokenizer=lambda x: x.split(),  # split on spaces
    lowercase=False,                # already lowercased
    stop_words=None                 # already removed
)

topic_model = BERTopic(
    min_topic_size=200,
    nr_topics=None,
    low_memory=True,
    calculate_probabilities=True,
    vectorizer_model=vectorizer_model,
    embedding_model=None
)

# Fit/transform using raw texts + reduced embeddings
topics_subset, probs_subset = topic_model.fit_transform(
    subset_texts, subset_embeddings_reduced
)
topics_full, probs_full = topic_model.transform(
    texts, full_embeddings_reduced
)

texts_df['topic'] = topics_full
if hasattr(probs_full, "shape") and len(probs_full.shape) > 1:
    texts_df['topic_confidence'] = [max(row) for row in probs_full]
else:
    texts_df['topic_confidence'] = probs_full

def describe_topic(model, topic_id, top_n=5):
    words = model.get_topic(topic_id)
    if not words:
        return "No description"
    return ", ".join([w[0] for w in words[:top_n]])

texts_df['topic_description'] = texts_df['topic'].apply(
    lambda t: describe_topic(topic_model, t)
)

end = datetime.now()
log_stage(conn, "BERTopic", start, end)


# 5. Sentiment analysis (GPU, batched)
start = datetime.now()
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
# pipeline device: 0 for GPU, -1 for CPU
pipeline_device = 0 if use_cuda else -1
if not use_cuda:
    print("CUDA not available — using CPU for sentiment pipeline")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer=tokenizer,
    device=pipeline_device
)

batch_size = 64
sentiments, confidences = [], []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    batch_sentiments = sentiment_analyzer(batch, truncation=True, max_length=512)
    sentiments.extend([s['label'] for s in batch_sentiments])
    confidences.extend([s['score'] for s in batch_sentiments])

texts_df['sentiment'] = sentiments
sentiment_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
texts_df['sentiment_description'] = texts_df['sentiment'].map(sentiment_map)
texts_df['sentiment_confidence'] = confidences
end = datetime.now()
log_stage(conn, "Sentiment Analysis", start, end)

# 6. Priority assignment (adjusted heuristic)
texts_df['priority'] = texts_df.apply(
    lambda row: "High" if (row['sentiment_description'] == "Negative" and (row['sentiment_confidence'] > 0.6 or row.get('topic_confidence', 0) > 0.6)) else "Normal",
    axis=1
)

# 7. Store outputs into SQLite table conversation_summary
conn.execute("""
    CREATE TABLE IF NOT EXISTS conversation_summary (
        conversation_id TEXT PRIMARY KEY,
        topic INTEGER,
        topic_confidence REAL,
        topic_description TEXT,
        sentiment TEXT,
        sentiment_confidence REAL,
        sentiment_description TEXT,
        priority TEXT
    )
""")

for _, row in texts_df.iterrows():
    conn.execute("""
        INSERT OR REPLACE INTO conversation_summary
        (conversation_id, topic, topic_confidence, topic_description,
         sentiment, sentiment_confidence, sentiment_description, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (row['conversation_id'], row['topic'], row['topic_confidence'],
          row['topic_description'], row['sentiment'], row['sentiment_confidence'],
          row['sentiment_description'], row['priority']))
conn.commit()


# 8. Write expanded confidence_metrics text file
with open("confidence_metrics.txt", "w") as f:
    f.write("Pipeline Confidence Metrics\n")
    f.write("===========================\n")
    f.write(f"Total conversations: {len(texts_df)}\n")
    f.write(f"Average topic confidence: {texts_df['topic_confidence'].mean():.3f}\n")
    f.write(f"Median topic confidence: {texts_df['topic_confidence'].median():.3f}\n")
    f.write(f"Average sentiment confidence: {texts_df['sentiment_confidence'].mean():.3f}\n")
    f.write(f"Median sentiment confidence: {texts_df['sentiment_confidence'].median():.3f}\n")
    f.write(f"High priority cases: {sum(texts_df['priority'] == 'High')}\n")
    f.write(f"Normal priority cases: {sum(texts_df['priority'] == 'Normal')}\n")
    f.write("\nTopic distribution (top 10):\n")
    for topic_id, count in texts_df['topic'].value_counts().head(10).items():
        description = describe_topic(topic_model, int(topic_id))
        f.write(f"{topic_id}: {description} ({count})\n")
    f.write("\nSentiment distribution:\n")
    for sentiment_label, count in texts_df['sentiment_description'].value_counts().items():
        f.write(f"{sentiment_label}: {count}\n")
    f.write("\nRun completed successfully.\n")

print("Pipeline completed at:", datetime.now())