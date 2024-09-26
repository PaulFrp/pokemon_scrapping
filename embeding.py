import pandas as pd
from sentence_transformers import SentenceTransformer

data = pd.read_csv("./data/filtered_file.csv")

model = SentenceTransformer("all-mpnet-base-v2")
print(data['comment'].isna().sum())
data['comment'] = data['comment'].fillna('')

data['embeddings'] = data['comment'].apply(lambda x: model.encode(x).tolist())

data.to_csv("./data/embeddings_data.csv", index=False)

print("Embeddings saved to embeddings_data.csv")
