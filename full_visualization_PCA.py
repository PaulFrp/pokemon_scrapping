import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

class DataVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.embeddings_normalized = None
        self.pca_result = None
        self.tsne_result = None
        self.umap_result = None
        self.kmeans = None

    def load_data(self):

        self.data = pd.read_csv(self.file_path)
        # Convert embeddings to numpy arrays
        self.data['embeddings'] = self.data['embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

    def normalize_embeddings(self):
        scaler = StandardScaler()
        self.embeddings_normalized = scaler.fit_transform(np.vstack(self.data['embeddings'].values))

    def apply_pca(self):
        pca = PCA(n_components=3)
        self.pca_result = pca.fit_transform(self.embeddings_normalized)

    def apply_tsne(self):
        tsne = TSNE(n_components=2, random_state=42)
        self.tsne_result = tsne.fit_transform(self.embeddings_normalized)

    def apply_umap(self):
        self.umap_result = umap.UMAP(n_components=2, random_state=42).fit_transform(self.embeddings_normalized)

    def plot_density_pca(self):
        pca_df = pd.DataFrame(self.pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
        plt.figure(figsize=(8, 6))
        sns.kdeplot(data=pca_df, x='PCA1', y='PCA2', cmap="Blues", fill=True, bw_adjust=0.5)
        plt.title('Density Plot of PCA Results')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.savefig('./graphs/density_pca.png')
        plt.show()

    def plot_violin_pca(self):
        pca_df = pd.DataFrame(self.pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=pca_df)
        plt.title('Violin Plot of PCA Components')
        plt.savefig('./graphs/violin_pca.png')
        plt.show()

    def plot_pairplot_pca(self):
        pca_df = pd.DataFrame(self.pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
        sns.pairplot(pca_df)
        plt.savefig('./graphs/pairplot_pca.png')
        plt.show()

    def plot_3d_pca(self):
        pca_df = pd.DataFrame(self.pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'], c='blue', alpha=0.5)
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_zlabel('PCA3')
        plt.title('3D PCA Scatter Plot')
        plt.savefig('./graphs/3d_pca_scatter.png')
        plt.show()

    def kmeans_clustering(self):
        pca_df = pd.DataFrame(self.pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
        self.kmeans = KMeans(n_clusters=5, random_state=42).fit(self.pca_result)
        pca_df['cluster'] = self.kmeans.labels_

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['cluster'], cmap='viridis')
        plt.title('K-Means Clustering on PCA')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.savefig('./graphs/kmeans_pca.png')
        plt.show()

    def generate_wordcloud(self):
        text = ' '.join(self.data['comment'].values)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Comments')
        plt.savefig('./graphs/wordcloud_comments.png')
        plt.show()

    def plot_silhouette(self):
        silhouette_vals = silhouette_samples(self.pca_result, self.kmeans.labels_)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(silhouette_vals)), silhouette_vals)
        plt.title('Silhouette Scores for K-Means Clustering')
        plt.savefig('./graphs/silhouette_kmeans.png')
        plt.show()

    def plot_similarity_heatmap(self):
        similarity_matrix = cosine_similarity(self.embeddings_normalized)
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='viridis')
        plt.title('Cosine Similarity Heatmap of Embeddings')
        plt.savefig('./graphs/similarity_heatmap.png')
        plt.show()

    def interactive_tsne_plot(self):
        tsne_df = pd.DataFrame(self.tsne_result, columns=['TSNE1', 'TSNE2'])
        fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2', hover_data=[self.data['comment']])
        fig.update_layout(title="Interactive t-SNE Visualization")
        fig.write_html("./graphs/tsne_plotly.html")
        fig.show()

    def plot_dendrogram(self):
        linked = linkage(self.pca_result, method='ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.title('Dendrogram for Hierarchical Clustering')
        plt.savefig('./graphs/dendrogram_clustering.png')
        plt.show()

    def run_all_visualizations(self):
        self.load_data()
        self.normalize_embeddings()
        self.apply_pca()
        self.apply_tsne()
        self.apply_umap()
        self.plot_density_pca()
        self.plot_violin_pca()
        self.plot_pairplot_pca()
        self.plot_3d_pca()
        self.kmeans_clustering()
        self.generate_wordcloud()
        self.plot_silhouette()
        self.plot_similarity_heatmap()
        self.interactive_tsne_plot()
        self.plot_dendrogram()

visualizer = DataVisualizer('./data/embedded/embeddings_merged_data.csv')
visualizer.run_all_visualizations()


