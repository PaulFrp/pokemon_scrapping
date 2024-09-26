import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import ast
from sklearn.preprocessing import StandardScaler

# Load your data
data = pd.read_csv('./data/embeddings_data.csv')  # Adjust the filename accordingly

# Ensure 'embeddings' are in a list format
data['embeddings'] = data['embeddings'].apply(ast.literal_eval).apply(np.array)

# Normalize the embeddings
scaler = StandardScaler()
embeddings_normalized = scaler.fit_transform(np.vstack(data['embeddings'].values))

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings_normalized)

# Apply UMAP
umap_results = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings_normalized)

# Create DataFrame for visualization
tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
umap_df = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])

# Plotting t-SNE results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], alpha=0.7)
plt.title('t-SNE Visualization')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')

# Plotting UMAP results
plt.subplot(1, 2, 2)
plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], alpha=0.7)
plt.title('UMAP Visualization')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')

# Show the plots
plt.tight_layout()
plt.savefig('./graphs/embedding_visualization.png')  # Save the plot as a file
plt.show()

# Output the number of points on the graph
print(f"Number of points in the visualization: {len(data)}")
