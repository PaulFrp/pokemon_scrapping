import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from wordcloud import STOPWORDS

class DataVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.embeddings_normalized = None
        self.umap_result = None
        self.kmeans = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        # Convert embeddings to numpy arrays
        self.data['embeddings'] = self.data['embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

    def normalize_embeddings(self):
        scaler = StandardScaler()
        self.embeddings_normalized = scaler.fit_transform(np.vstack(self.data['embeddings'].values))

    def apply_umap(self):
        self.umap_result = umap.UMAP(n_components=2, random_state=42).fit_transform(self.embeddings_normalized)

    def plot_density_umap(self):
       
        umap_df = pd.DataFrame(self.umap_result, columns=['UMAP1', 'UMAP2'])
        plt.figure(figsize=(8, 6))
        sns.kdeplot(data=umap_df, x='UMAP1', y='UMAP2', cmap="Blues", fill=True, bw_adjust=0.5)
        plt.title('Density Plot of UMAP Results')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.savefig('./graphs/density_umap.png')
        plt.show()

    def plot_violin_umap(self):
        
        umap_df = pd.DataFrame(self.umap_result, columns=['UMAP1', 'UMAP2'])
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=umap_df)
        plt.title('Violin Plot of UMAP Components')
        plt.savefig('./graphs/violin_umap.png')
        plt.show()

    def plot_pairplot_umap(self):
        
        umap_df = pd.DataFrame(self.umap_result, columns=['UMAP1', 'UMAP2'])
        sns.pairplot(umap_df)
        plt.savefig('./graphs/pairplot_umap.png')
        plt.show()

    def kmeans_clustering(self):
       
        self.kmeans = KMeans(n_clusters=4, random_state=42).fit(self.umap_result)
        umap_df = pd.DataFrame(self.umap_result, columns=['UMAP1', 'UMAP2'])
        umap_df['cluster'] = self.kmeans.labels_

        plt.figure(figsize=(8, 6))
        plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c=umap_df['cluster'], cmap='viridis')
        plt.title('K-Means Clustering on UMAP')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.savefig('./graphs/kmeans_umap.png')
        plt.show()

    def generate_wordcloud(self):

        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(["Pokemon", "one", "game", "Pokmon", "think", "thing"])

        text = ' '.join(self.data['comment'].values)
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Comments')
        plt.savefig('./graphs/wordcloud_comments.png')
        plt.show()

    def plot_silhouette(self):
      
        silhouette_vals = silhouette_samples(self.umap_result, self.kmeans.labels_)
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

    def interactive_umap_plot(self):
        
        umap_df = pd.DataFrame(self.umap_result, columns=['UMAP1', 'UMAP2'])
        fig = px.scatter(umap_df, x='UMAP1', y='UMAP2', hover_data=[self.data['comment']])
        fig.update_layout(title="Interactive UMAP Visualization")
        fig.write_html("./graphs/umap_plotly.html")
        fig.show()

    def plot_dendrogram(self):
        
        linked = linkage(self.umap_result, method='ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.title('Dendrogram for Hierarchical Clustering')
        plt.savefig('./graphs/dendrogram_clustering.png')
        plt.show()

    def elbow_method(self, max_clusters=10):
        inertia = []
        cluster_range = range(1, max_clusters + 1)
    
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.umap_result) 
            inertia.append(kmeans.inertia_)
    
        plt.figure(figsize=(8, 6))
        plt.plot(cluster_range, inertia, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.xticks(cluster_range)
        plt.grid(True)
        plt.savefig('./graphs/elbow_method.png')
        plt.show()



    def calculate_cluster_centroids(self):
        # Extract the cluster centroids from K-Means
        centroids = self.kmeans.cluster_centers_
        return centroids

    def find_closest_comments(self):
        # Centroids in the original 768-dimensional space, not the UMAP-reduced space
        centroids = np.vstack([self.embeddings_normalized[self.kmeans.labels_ == i].mean(axis=0) for i in range(self.kmeans.n_clusters)])

    # Cosine similarity between centroids and embeddings
        similarity_matrix = cosine_similarity(centroids, self.embeddings_normalized)

    # Find the closest comments for each centroid
        closest_comments = []
        for i in range(self.kmeans.n_clusters):
            closest_idx = np.argmax(similarity_matrix[i])
            closest_comments.append(self.data['comment'].iloc[closest_idx])

        return closest_comments


    def wordcloud_for_clusters(self):
        closest_comments = self.find_closest_comments()

        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(["Pokemon", "one", "game", "Pokmon", "think", "thing", "things", "Thank", "Thanks", "games", "see", "time", "something", "way", "make", "use", "even", "Im", "S", "Pokémon", "really", "new"])

        for i, comment in enumerate(closest_comments):
        # Generate a word cloud for each cluster based on comments
            text = ' '.join(self.data['comment'][self.kmeans.labels_ == i].values)
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(text)
        
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for Cluster {i+1}')
            plt.savefig(f'./graphs/wordcloud_cluster_{i+1}.png')
            plt.show()


    def run_all_visualizations(self):
        self.load_data()
        self.normalize_embeddings()
        self.apply_umap()
        #self.plot_density_umap()
        #self.plot_violin_umap()
        #self.plot_pairplot_umap()
        self.kmeans_clustering()
        self.generate_wordcloud()
        #self.plot_silhouette()
        #self.elbow_method(max_clusters = 20)
        #self.plot_similarity_heatmap()
        #self.interactive_umap_plot()
        #self.plot_dendrogram()

        # Generate word clouds for each cluster's comments
        self.wordcloud_for_clusters()

# Example usage:
visualizer = DataVisualizer('./data/embedded/embeddings_merged_data.csv')
visualizer.run_all_visualizations()

