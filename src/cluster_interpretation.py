import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

def analyze_clusters(df_clustered, output_path):

    df = df_clustered.copy()
    df['category'] = df['categories_tags'].str.split(',')
    df_exp = df.explode('category')


    cat_cluster_counts = (
        df_exp
        .groupby(['category', 'cluster_id'])
        .size()
        .unstack(fill_value=0)
    )

    cat_total_counts = cat_cluster_counts.sum(axis=1)

    cat_cluster_pct = (
        cat_cluster_counts
        .div(cat_total_counts, axis=0) * 100
    )

    cat_cluster_pct_sorted = cat_cluster_pct.loc[
        cat_total_counts.sort_values(ascending=False).index
    ]
    cat_cluster_pct_sorted['total_count'] = cat_total_counts.loc[
        cat_cluster_pct_sorted.index
    ]

    nova_cluster_counts = (
        df
        .groupby(['nova_group', 'cluster_id'])
        .size()
        .unstack(fill_value=0)
    )

    nova_cluster_pct = (
        nova_cluster_counts
        .div(nova_cluster_counts.sum(axis=1), axis=0) * 100
    )

    cluster_sizes = (
        df.groupby('cluster_id')
        .size()
        .to_frame()
        .T
    )
    cluster_sizes.index = ['count']

    cluster_cat_counts = (
        df_exp
        .groupby(['cluster_id', 'category'])
        .size()
        .unstack(fill_value=0)
    )

    cluster_cat_pct = (
        cluster_cat_counts
        .div(cluster_cat_counts.sum(axis=1), axis=0) * 100
    )

    cluster_nova_counts = (
        df
        .groupby(['cluster_id', 'nova_group'])
        .size()
        .unstack(fill_value=0)
    )

    cluster_nova_pct = (
        cluster_nova_counts
        .div(cluster_nova_counts.sum(axis=1), axis=0) * 100
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        

        cat_cluster_pct_sorted.to_excel(writer, sheet_name="category_cluster_pct")
        

        nova_cluster_pct.to_excel(writer, sheet_name="nova_cluster_pct")
        

        cluster_sizes.to_excel(writer, sheet_name="cluster_sizes")
        

        cluster_cat_pct.to_excel(writer, sheet_name="cluster_cat_pct")
        

        cluster_nova_pct.to_excel(writer, sheet_name="cluster_nova_pct")

    print("Saved to:", output_path)

def plot_category_tail(cluster_df, cutoff=95):

    df_exp = cluster_df.copy()
    df_exp['category'] = df_exp['categories_tags'].str.split(',')
    df_exp = df_exp.explode('category')
    
    # Count categories
    cat_counts = df_exp['category'].value_counts()
    cat_counts_df = pd.DataFrame(cat_counts).reset_index()
    cat_counts_df.columns = ['category', 'count']
    
    # Percent and cumulative percent
    total_count = cat_counts_df['count'].sum()
    cat_counts_df = cat_counts_df.sort_values('count', ascending=True)  # smallest to largest
    cat_counts_df['percent'] = cat_counts_df['count'] / total_count * 100
    cat_counts_df['cumulative_percent'] = cat_counts_df['percent'].cumsum()
    
    # Find cutoff index
    cutoff_idx = cat_counts_df[cat_counts_df['cumulative_percent'] > cutoff].index[0]

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(cat_counts_df['cumulative_percent'].values, color='steelblue')
    # plt.axvline(x=cutoff_idx, color='red', linestyle='--', label=f'{cutoff}% cutoff')
    plt.xlabel("Categories (sorted by count, ascending)")
    plt.ylabel("Cumulative Percentage of Products (%)")
    plt.title("Category Coverage - Long Tail Visualization")
    # plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    return cat_counts_df

# # Example usage
#     cat_counts_df = plot_category_tail(cluster_df_dist_cats_meta, cutoff=95)


def find_tail(cluster_df, cutoff):
    df_exp = cluster_df.copy()
    df_exp['category'] = df_exp['categories_tags'].str.split(',')
    df_exp = df_exp.explode('category')
    cat_counts = df_exp['category'].value_counts()
    cat_counts_df = pd.DataFrame(cat_counts).reset_index()
    cat_counts_df.columns = ['category', 'count']
    total_count = cat_counts_df['count'].sum()
    cat_counts_df['percent'] = cat_counts_df['count'] / total_count * 100
    cat_counts_df = cat_counts_df.sort_values('count', ascending=True)
    cat_counts_df['cumulative_percent'] = cat_counts_df['percent'].cumsum()
    keep_cats = cat_counts_df[cat_counts_df['cumulative_percent'] >cutoff]['category'].tolist()
    keep_len = len(keep_cats)
    old_len = cat_counts_df.shape[0]
    print(f"{keep_len} categories are more than {100-cutoff}% out of {old_len}")
    print(f"{old_len - keep_len} categories are less than {cutoff}% of the entire data and should be removed")
    plot_category_tail(cluster_df, cutoff)
    return keep_cats


def keep_frequent_categories(cluster_df, keep_cats):
    def has_frequent_category(cat_string):
        if pd.isna(cat_string) or cat_string == '':
            return False
        cats = cat_string.split(',')
        return any(cat in keep_cats for cat in cats)

    # Filter products
    df_filtered = cluster_df[cluster_df['categories_tags'].apply(has_frequent_category)].copy()

    print(f"Original: {len(cluster_df)} products")
    print(f"Filtered: {len(df_filtered)} products ({len(df_filtered)/len(cluster_df)*100:.1f}%)")
    return df_filtered

# def recommend_alternates(df, product_name, product_nova, topN=3, isSameCluster=True):

    

#     df = df.copy()
#     df['category_list'] = df['categories_tags'].fillna('').apply(lambda x: [c.strip() for c in x.split(',')])
    

#     product_row = df[df['product_name'] == product_name]
    
#     if product_row.empty:
#         return "Sorry, we couldn't find any suitable alternates. We'll work on improving our recommendations!"
    
#     product_row = product_row.iloc[0]
#     user_categories = set(product_row['category_list'])
#     user_cluster = product_row['cluster_id']
    

#     candidates = df[df['product_name'] != product_name].copy()
    
#     if isSameCluster:
#         candidates = candidates[candidates['cluster_id'] == user_cluster]
    

#     candidates = candidates[candidates['category_list'].apply(lambda cats: len(user_categories.intersection(cats)) > 0)]
    

#     healthier_candidates = candidates[candidates['nova_group'] < product_nova]
    
#     if not healthier_candidates.empty:
#         healthier_candidates = healthier_candidates.sort_values('nova_group') 
#         top_recs = list(zip(healthier_candidates['product_name'], healthier_candidates['nova_group']))[:topN]
#         return top_recs
    

#     if product_nova == 1:
#         same_nova_candidates = candidates[candidates['nova_group'] == 1]
#         if not same_nova_candidates.empty:
#             top_recs = list(zip(same_nova_candidates['product_name'], same_nova_candidates['nova_group']))[:topN]
#             return [("You have a very least-processed food but here is an alternate:")] + top_recs
    

#     same_nova_candidates = candidates[candidates['nova_group'] == product_nova]
#     if not same_nova_candidates.empty:
#         top_recs = list(zip(same_nova_candidates['product_name'], same_nova_candidates['nova_group']))[:topN]
#         return [("Couldn't find less processed options. Try these with same NOVA:", None)] + top_recs
    

#     return "Sorry, we couldn't find any suitable alternates. We'll work on improving our recommendations!"



