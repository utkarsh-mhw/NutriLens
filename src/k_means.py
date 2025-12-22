import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def train_test_split_stratified(df, test_size=0.1, random_state=42):

    train_df = df.groupby('nova_group', group_keys=False).apply(
        lambda x: x.sample(frac=(1-test_size), random_state=random_state)
    ).reset_index(drop=True)
    
    test_df = df[~df.index.isin(train_df.index)].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, test_df

def extract_embeddings(df, start_col=18):
    ## assumes first 3 columsna re metadata, change this if you change the data format
    
    exclude_cols = ['code', 'additive_density', 'allergens_en']
    # Extract metadata
    meta_cols = df.columns[:start_col].tolist()
    # meta_cols = [col for col in meta_cols if col not in exclude_cols]
    df_meta = df[meta_cols].copy()
    
    # Extract embeddings
    embedding_cols = df.columns[start_col:].tolist()
    # embedding_cols = [col for col in embedding_cols if col not in exclude_cols]
    embeddings = df[embedding_cols].values
    
    return embeddings, df_meta


def reduce_dimensions_umap(embeddings, n_components=50, random_state=42):

    print(f"Reducing from {embeddings.shape[1]} to {n_components} dimensions using UMAP")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=random_state,
        verbose=True
    )
    
    embeddings_reduced = reducer.fit_transform(embeddings)
    
    print(f"UMAP reduction complete: {embeddings_reduced.shape}")
    
    return embeddings_reduced, reducer


def cluster_kmeans(features, n_clusters=30, use_minibatch=False, random_state=42):

    print(f"K-Means: k={n_clusters}")
    
    if use_minibatch:
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, 
                                batch_size=2048, n_init=10)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    
    labels = model.fit_predict(features)
    
    print(f"Cluster sizes: min={np.bincount(labels).min()}, max={np.bincount(labels).max()}")
    
    return model, labels


def eval_clustering(features, labels):
    """Calculate clustering metrics"""
    sil = silhouette_score(features, labels)
    db = davies_bouldin_score(features, labels)
    ch = calinski_harabasz_score(features, labels)
    
    print(f"\nMetrics:")
    print(f"  Silhouette: {sil:.4f}")
    print(f"  Davies-Bouldin: {db:.4f}")
    print(f"  Calinski-Harabasz: {ch:.2f}")
    
    return {'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch}


def run_kmeans_pipeline(df, n_clusters=30, n_components=150, use_minibatch=True):

    
    # Split data
    train_df, test_df = train_test_split_stratified(df, test_size=0.1)
    
    # Extract embeddings
    train_emb, train_meta = extract_embeddings(train_df)
    
    # UMAP
    train_reduced, reducer = reduce_dimensions_umap(train_emb, n_components=n_components)
    
    # K-Means
    model, labels = cluster_kmeans(train_reduced, n_clusters=n_clusters, 
                                    use_minibatch=use_minibatch)
    
    # Add labels to dataframe
    train_meta['cluster_id'] = labels
    
    # Evaluate
    metrics = eval_clustering(train_reduced, labels)
    
    return train_meta, test_df, model, reducer, metrics