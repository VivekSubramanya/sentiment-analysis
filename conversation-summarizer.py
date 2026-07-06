import pandas
import sqlite3

conn = sqlite3.connect('twcs_processed.db')

data = pandas.read_sql('select * from tweets',conn)

def get_root(tweet_id=str, tweet_list=[]):
    parent_id = tweet_list.get(tweet_id)
    if parent_id is None or pandas.isna(parent_id) or parent_id == 'nan':
        return tweet_id
    else:
        return get_root(parent_id,tweet_list)
    
def assign_conversation_id(data):
    parent_map = dict(zip(data['tweet_id'].astype(str),data['in_response_to_tweet_id'].astype(str).replace('NaN',None)))
    data['conversation_id'] = data['tweet_id'].astype(str).apply(lambda tid: get_root(tid,parent_map))
    return data

df = assign_conversation_id(data)
df['conversation_id'] = df['conversation_id'].str.split('.').str[0]
df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].str.split('.').str[0]
# print(df)

cursor = conn.cursor()
cursor.execute("pragma table_info(tweets);")
columns = [row[1] for row in cursor.fetchall()]

if 'conversation_id' not in columns:
    conn.execute('alter table tweets add column conversation_id text')

df.to_sql('tweets',conn,if_exists="replace", index=False)