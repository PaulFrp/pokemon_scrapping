import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load your data
data = pd.read_csv('./data/embeddings_data.csv')

# Convert the embeddings column to a list of arrays
data['embeddings'] = data['embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(list(data['embeddings']))

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])


# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], alpha=0.5)

num_points = len(pca_df)
plt.text(0.05, 0.95, f'Number of Points: {num_points}', fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)

plt.title('PCA of Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid()

plt.savefig('pca_embeddings_plot.png', dpi=300, bbox_inches='tight') 

plt.show()
