import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)

origianl_df = pd.read_csv('C:/Projects/Data/clustering_data.csv')

wss_list = []

for k in range(1, 11):
    df = origianl_df.copy()
    def Euclidean_distance(row, centroid):
        closest_centroid = None
        min = float('inf')
        for i in centroid.keys():
            distance = np.sqrt(np.sum((row.values-centroid[i])**2))
            if distance < min:
                min = distance 
                closest_centroid = i
        return closest_centroid

    df['cluster'] = np.random.randint(1, k + 1, size=len(df))

    iteration = 0
    while True:
        iteration += 1
        
        prev_clusters = df['cluster'].tolist()
        
        centroid = {
            i: df[df['cluster'] == i].drop(columns='cluster').mean().values
            for i in df['cluster'].unique()
        }
        
        cluster_list = []
        for row in df.drop(columns='cluster').iterrows():
            cluster_list.append(Euclidean_distance(row[1], centroid))
        df['cluster'] = cluster_list
        
        if cluster_list == prev_clusters:
            print(f"\nConverged in {iteration} iterations.")
            break

    # Calculate Within-Cluster Sum of Squares (WSS)
    wss = 0
    for i in df['cluster'].unique():
        cluster_points = df[df['cluster'] == i].drop(columns = 'cluster').values
        center = centroid[i] 
        distances = np.sum((cluster_points - center) ** 2)
        wss += distances
    wss_list.append((k, wss))

sklearn_wss_list = []

features = origianl_df.drop(columns='cluster', errors='ignore')

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(features)
    sklearn_wss_list.append((k, model.inertia_))  

ks_custom, wss_custom = zip(*wss_list)
ks_sklearn, wss_sklearn = zip(*sklearn_wss_list)

print(wss_list)
print(sklearn_wss_list)

plt.figure(figsize=(10, 6))
plt.plot(ks_custom, wss_custom, marker='o', label='Custom KMeans')
plt.plot(ks_sklearn, wss_sklearn, marker='x', linestyle='--', label='sklearn KMeans')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WSS)")
plt.title("Elbow Method: Custom vs sklearn KMeans")
plt.legend()
plt.grid(True)
plt.show()

    


