import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_clustering(features, labels, algorithm_name="K-Means"):
    """
    Compute internal validation metrics for clustering
    
    Parameters:
    -----------
    features : array, reduced embeddings used for clustering
    labels : array, cluster assignments
    algorithm_name : str, name for display
    
    Returns:
    --------
    metrics : dict with all computed metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING: {algorithm_name}")
    print(f"{'='*60}")
    
    n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise for DBSCAN
    
    if n_clusters < 2:
        print("WARNING: Less than 2 clusters found. Cannot compute metrics.")
        return None
    
    metrics = {
        'algorithm': algorithm_name,
        'n_clusters': n_clusters,
        'n_samples': len(labels)
    }
    
    # Silhouette Score (range: -1 to 1, higher is better)
    # Measures how similar points are to their own cluster vs other clusters
    try:
        metrics['silhouette'] = silhouette_score(features, labels)
        print(f"✓ Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"  (Range: -1 to 1, higher is better)")
    except Exception as e:
        print(f"✗ Could not compute Silhouette Score: {e}")
        metrics['silhouette'] = None
    
    # Davies-Bouldin Index (range: 0 to ∞, lower is better)
    # Measures average similarity between each cluster and its most similar cluster
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
        print(f"✓ Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
        print(f"  (Range: 0 to ∞, lower is better)")
    except Exception as e:
        print(f"✗ Could not compute Davies-Bouldin Index: {e}")
        metrics['davies_bouldin'] = None
    
    # Calinski-Harabasz Score (range: 0 to ∞, higher is better)
    # Ratio of between-cluster to within-cluster variance
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)
        print(f"✓ Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}")
        print(f"  (Range: 0 to ∞, higher is better)")
    except Exception as e:
        print(f"✗ Could not compute Calinski-Harabasz Score: {e}")
        metrics['calinski_harabasz'] = None
    
    # Cluster size distribution
    unique, counts = np.unique(labels, return_counts=True)
    metrics['min_cluster_size'] = counts.min()
    metrics['max_cluster_size'] = counts.max()
    metrics['mean_cluster_size'] = counts.mean()
    metrics['std_cluster_size'] = counts.std()
    
    print(f"\nCluster Size Statistics:")
    print(f"  Min: {metrics['min_cluster_size']}")
    print(f"  Max: {metrics['max_cluster_size']}")
    print(f"  Mean: {metrics['mean_cluster_size']:.1f}")
    print(f"  Std: {metrics['std_cluster_size']:.1f}")
    
    print(f"{'='*60}\n")
    
    return metrics


def compare_algorithms(results_list, output_dir='outputs/plots'):
    """
    Compare multiple clustering algorithms
    
    Parameters:
    -----------
    results_list : list of dicts from evaluate_clustering()
    output_dir : str, directory to save plots
    
    Returns:
    --------
    comparison_df : DataFrame with all metrics
    """
    # Filter out None results
    results_list = [r for r in results_list if r is not None]
    
    if not results_list:
        print("No valid results to compare")
        return None
    
    comparison_df = pd.DataFrame(results_list)
    
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")
    
    # Create comparison plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    metrics_to_plot = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    metrics_available = [m for m in metrics_to_plot if m in comparison_df.columns and comparison_df[m].notna().any()]
    
    if not metrics_available:
        print("No metrics available to plot")
        return comparison_df
    
    n_metrics = len(metrics_available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = sns.color_palette("Set2", len(comparison_df))
    
    for idx, metric in enumerate(metrics_available):
        ax = axes[idx]
        
        # Bar plot
        bars = ax.bar(comparison_df['algorithm'], comparison_df[metric], color=colors)
        
        # Formatting
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Rotate x labels if needed
        if len(comparison_df) > 3:
            ax.set_xticklabels(comparison_df['algorithm'], rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'algorithm_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    plt.close()
    
    return comparison_df


def silhouette_analysis_by_k(features, k_range=range(20, 150, 10), random_state=42, output_dir='outputs/plots'):
    """
    Compute silhouette scores for different k values
    Helps determine optimal number of clusters
    
    Parameters:
    -----------
    features : array, reduced embeddings
    k_range : range or list, k values to test
    random_state : int
    output_dir : str
    
    Returns:
    --------
    k_values : list
    silhouette_scores : list
    """
    from sklearn.cluster import KMeans
    
    print(f"\nRunning silhouette analysis for k in {list(k_range)}...")
    
    silhouette_scores = []
    
    for k in k_range:
        print(f"  Testing k={k}...", end=' ')
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
        print(f"Silhouette: {score:.4f}")
    
    # Plot
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'silhouette_vs_k.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Silhouette analysis plot saved to: {output_path}")
    plt.close()
    
    return list(k_range), silhouette_scores


def combined_k_analysis(features, k_range=range(20, 150, 10), random_state=42, output_dir='outputs/plots'):
    """
    Combined elbow and silhouette analysis
    
    Parameters:
    -----------
    features : array
    k_range : range or list
    random_state : int
    output_dir : str
    
    Returns:
    --------
    results_df : DataFrame with inertia and silhouette for each k
    """
    from sklearn.cluster import KMeans
    
    print(f"\n{'='*60}")
    print(f"COMBINED K ANALYSIS")
    print(f"{'='*60}")
    print(f"Testing k values: {list(k_range)}")
    
    inertias = []
    silhouettes = []
    
    for k in k_range:
        print(f"\nk={k}:")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(features)
        
        inertia = kmeans.inertia_
        silhouette = silhouette_score(features, labels)
        
        inertias.append(inertia)
        silhouettes.append(silhouette)
        
        print(f"  Inertia: {inertia:.2f}")
        print(f"  Silhouette: {silhouette:.4f}")
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'k': list(k_range),
        'inertia': inertias,
        'silhouette': silhouettes
    })
    
    # Save to CSV
    # os.makedirs(output_dir, exist_ok=True)
    # csv_path = os.path.join(output_dir.replace('plots', 'results'), 'k_analysis_results.csv')
    # os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # results_df.to_csv(csv_path, index=False)
    # print(f"\n✓ Results saved to: {csv_path}")
    
    # Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow plot
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # output_path = os.path.join(output_dir, 'combined_k_analysis.png')
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # print(f"✓ Combined plot saved to: {output_path}")
    plt.close()
    
    print(f"{'='*60}\n")
    
    return results_df