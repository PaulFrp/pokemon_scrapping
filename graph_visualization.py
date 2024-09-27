import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import ast
from sklearn.preprocessing import StandardScaler

class CombinedEmbeddingVisualizer:
    def __init__(self, file_path, pca_output_image_path, tsne_umap_output_image_path):
        self.file_path = file_path
        self.pca_output_image_path = pca_output_image_path
        self.tsne_umap_output_image_path = tsne_umap_output_image_path
        self.data = None
        self.embeddings_normalized = None
        self.pca_result = None
        self.tsne_results = None
        self.umap_results = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        # Ensure 'embeddings' are in a list format
        self.data['embeddings'] = self.data['embeddings'].apply(ast.literal_eval).apply(np.array)

    def normalize_embeddings(self):
        scaler = StandardScaler()
        self.embeddings_normalized = scaler.fit_transform(np.vstack(self.data['embeddings'].values))

    def apply_pca(self):
        pca = PCA(n_components=2)
        self.pca_result = pca.fit_transform(self.embeddings_normalized)

    def plot_pca(self):
        pca_df = pd.DataFrame(self.pca_result, columns=['PCA1', 'PCA2'])
        num_points = len(pca_df)

        plt.figure(figsize=(12, 8))
        plt.scatter(pca_df['PCA1'], pca_df['PCA2'], alpha=0.5)
        plt.text(0.05, 0.95, f'Number of Points: {num_points}', fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)

        plt.title('PCA of Embeddings')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid()

        plt.savefig(self.pca_output_image_path, dpi=300, bbox_inches='tight')
        plt.show()

    def apply_tsne(self):
        tsne = TSNE(n_components=2, random_state=42)
        self.tsne_results = tsne.fit_transform(self.embeddings_normalized)

    def apply_umap(self):
        self.umap_results = umap.UMAP(n_components=2, random_state=42).fit_transform(self.embeddings_normalized)

    def plot_tsne_umap(self):
        tsne_df = pd.DataFrame(self.tsne_results, columns=['TSNE1', 'TSNE2'])
        umap_df = pd.DataFrame(self.umap_results, columns=['UMAP1', 'UMAP2'])

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], alpha=0.7)
        plt.title('t-SNE Visualization')
        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')

        plt.subplot(1, 2, 2)
        plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], alpha=0.7)
        plt.title('UMAP Visualization')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')

        plt.tight_layout()
        plt.savefig(self.tsne_umap_output_image_path, dpi=300)
        plt.show()

        print(f"Number of points in the visualization: {len(self.data)}")

    def run_pca(self):
        """Run PCA-related steps"""
        self.load_data()
        self.normalize_embeddings()
        self.apply_pca()
        self.plot_pca()

    def run_tsne_umap(self):
        """Run t-SNE and UMAP-related steps"""
        self.load_data()
        self.normalize_embeddings()
        self.apply_tsne()
        self.apply_umap()
        self.plot_tsne_umap()

visualizer = CombinedEmbeddingVisualizer(
    file_path='./data/embedded/embeddings_merged_data.csv',
    pca_output_image_path='./graphs/pca_graph_merged_full_cleaned.png',
    tsne_umap_output_image_path='./graphs/tsne_umap_graph_merged_full_cleaned.png'
)

visualizer.run_pca()

visualizer.run_tsne_umap()
