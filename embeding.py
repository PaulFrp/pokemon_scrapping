import pandas as pd
from sentence_transformers import SentenceTransformer

class EmbeddingsGenerator:
    def __init__(self, input_file, output_file, model_name="all-mpnet-base-v2"):
        self.input_file = input_file
        self.output_file = output_file
        self.model = SentenceTransformer(model_name)

    def load_data(self):
        self.data = pd.read_csv(self.input_file)
        print(f"Number of missing comments: {self.data['comment'].isna().sum()}")
        self.data['comment'] = self.data['comment'].fillna('')

    def generate_embeddings(self):
        self.data['embeddings'] = self.data['comment'].apply(lambda x: self.model.encode(x).tolist())

    def save_data(self):
        self.data.to_csv(self.output_file, index=False)
        print(f"Embeddings saved to {self.output_file}")

    def run(self):
        self.load_data()
        self.generate_embeddings()
        self.save_data()

generator = EmbeddingsGenerator(input_file="./data/cleaned/merged_filtered_output.csv", output_file="./data/embedded/embeddings_merged_data.csv")
generator.run()
