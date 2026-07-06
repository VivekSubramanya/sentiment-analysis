import spacy
import re
import pandas as pd
import sqlite3
from multiprocessing import Pool
from datetime import datetime
from spacy.language import Language
from spacy.tokens import Doc

# Regex for emojis
builtin_emoji_pattern = re.compile(
    r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F900-\U0001F9FF]'
)

# Custom components
@Language.component("remove_mentions")
def remove_mentions(doc):
    cleaned_text = doc.text
    cleaned_text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'@\w+', '<MENTION>', cleaned_text)
    cleaned_text = re.sub(r'#\w+', '<HASHTAG>', cleaned_text)
    cleaned_text = re.sub(builtin_emoji_pattern, '<EMOJI>', cleaned_text)
    doc._.cleaned_text = cleaned_text
    return doc

@Language.component("anonymize_text")
def anonymize_text(doc):
    cleaned_text = doc._.cleaned_text
    for ent in doc.ents:
        cleaned_text = cleaned_text.replace(ent.text, f"<{ent.label_}>")
    doc._.cleaned_text = cleaned_text
    return doc

def init_spacy():
    Doc.set_extension("cleaned_tokens", default=[],force=True)
    Doc.set_extension("cleaned_text", default="",force=True)
    nlp = spacy.load("en_core_web_sm", exclude=["parser","ner","attr_ruler","tok2vec"])
    nlp.add_pipe("remove_mentions", first=True)
    nlp.add_pipe("anonymize_text", last=True)
    return nlp

def preprocess_text(text, nlp):
    doc = nlp(text)
    cleaned_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if re.match(r'[^\w\s]', token.text):
            continue
        if token.like_url or token.like_email:
            continue
        cleaned_tokens.append(token.text.lower())
    doc._.cleaned_tokens = cleaned_tokens
    return {"tokens": " ".join(doc._.cleaned_tokens), "cleaned_text": doc._.cleaned_text}

# Worker: preprocess chunk, return DataFrame (no DB writes!)
def process_chunk(args):
    chunk, idx = args
    nlp = init_spacy()
    chunk = chunk.astype({'tweet_id': str, 'in_response_to_tweet_id': str, 'response_tweet_id': str})
    chunk[['tokens','cleaned_text']] = chunk['text'].apply(lambda t: preprocess_text(t, nlp)).apply(pd.Series)
    return chunk

def parallel_process_to_sqlite(input_file, db_file="twcs_processed.db", table_name="tweets",
                               chunksize=10000, workers=4):
    # Initialize SQLite schema once
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            tweet_id TEXT,
            author_id TEXT,
            inbound BOOLEAN,
            created_at TEXT,
            text TEXT,
            response_tweet_id TEXT,
            in_response_to_tweet_id TEXT,
            tokens TEXT,
            cleaned_text TEXT
        )
    """)
    conn.commit()

    reader = pd.read_csv(input_file, header=0, chunksize=chunksize, encoding='utf-8', on_bad_lines='skip')

    data_processed = 0
    with Pool(workers) as pool:
        for processed_chunk in pool.imap_unordered(process_chunk, [(chunk, i) for i, chunk in enumerate(reader)]):
            # Only main process writes to SQLite
            if data_processed == 0:
                processed_chunk.to_sql(table_name, conn, if_exists="replace", index=False)
            else:
                processed_chunk.to_sql(table_name, conn, if_exists="append", index=False)
            data_processed += len(processed_chunk)
            print("processed", data_processed, "rows so far")

    conn.close()

if __name__ == "__main__":
    print("start time:", datetime.now())
    parallel_process_to_sqlite("twcs.csv", chunksize=10000, workers=8)
    print("end time:", datetime.now())