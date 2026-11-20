from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ROOT = Path(r"D:\\KRB_FINAL_GRAPH_PROJECT_V6")
EMBEDDINGS = ROOT / "VGAE_BASELINE" / "V6_VGAE_embeddings.npy"
OUT = ROOT / "VGAE_BASELINE"
FINAL_K = 10
SEED = 42

z = np.load(EMBEDDINGS)
if z.shape != (508, 16):
    raise ValueError(f"Expected embeddings (508, 16), found {z.shape}")
rows=[]
labels_final=None
for k in range(2,17):
    model=KMeans(n_clusters=k,n_init=30,random_state=SEED)
    labels=model.fit_predict(z)
    rows.append({"K":k,"silhouette":silhouette_score(z,labels),"davies_bouldin":davies_bouldin_score(z,labels),"calinski_harabasz":calinski_harabasz_score(z,labels),"min_cluster_size":int(pd.Series(labels).value_counts().min()),"max_cluster_size":int(pd.Series(labels).value_counts().max())})
    if k==FINAL_K: labels_final=labels
pd.DataFrame(rows).to_csv(OUT/"VGAE_clustering_metrics_K2_K16.csv",index=False)
pd.DataFrame({"node_index":np.arange(508),"cluster_K10":labels_final}).to_csv(OUT/"VGAE_cluster_assignments_K10.csv",index=False)
print(f"Saved clustering metrics and final K={FINAL_K} assignments")
