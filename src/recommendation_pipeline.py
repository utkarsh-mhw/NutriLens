import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import umap

from k_means import *


def predict_cluster(demo_product):
    
    # open models and reducer
    models_dir = '../models/'
    with open(f'{models_dir}umap_reducer.pkl', 'rb') as f:
            umap_reducer = pickle.load(f)

    data_dir = '../data/'
    with open(f'{models_dir}kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)

    product_embeddings, product_meta = extract_embeddings(demo_product)
    embeddings_reduced = umap_reducer.transform(product_embeddings)
    cluster_ids = kmeans_model.predict(embeddings_reduced)
    return cluster_ids[0]


    
def fetch_user_product_details(product_name):
    temp_df = pd.read_csv("../data/analysis_data/demo_data.csv", index_col=0)
    lookup_product = temp_df[temp_df['product_name'] == product_name]
    return lookup_product


def recommend_alternates(product_name, product_nova, topN=3, isSameCluster=True):
    
    product_row = fetch_user_product_details(product_name)
    product_cluster_id = predict_cluster(product_row)
    product_row['cluster_id'] = product_cluster_id
    product_row['category_list'] = product_row['categories_tags'].fillna('').apply(lambda x: [c.strip() for c in x.split(',')])

    df = pd.read_csv("../data/analysis_data/kmeans_output_final.csv")
    df = df.copy()
    df['category_list'] = df['categories_tags'].fillna('').apply(lambda x: [c.strip() for c in x.split(',')])
    


    # product_row = df[df['product_name'] == product_name]
    
    if product_row.empty:
        return "Sorry, we couldn't find any suitable alternates. We'll work on improving our recommendations!"
    
    product_row = product_row.iloc[0]
    user_categories = set(product_row['category_list'])
    user_cluster = product_row['cluster_id']
    

    candidates = df[df['product_name'] != product_name].copy()
    
    if isSameCluster:
        candidates = candidates[candidates['cluster_id'] == user_cluster]
    

    candidates = candidates[candidates['category_list'].apply(lambda cats: len(user_categories.intersection(cats)) > 0)]
    

    healthier_candidates = candidates[candidates['nova_group'] < product_nova]
    
    if not healthier_candidates.empty:
        healthier_candidates = healthier_candidates.sort_values('nova_group') 
        top_recs = list(zip(healthier_candidates['product_name'], healthier_candidates['nova_group']))[:topN]
        return top_recs
    

    if product_nova == 1:
        same_nova_candidates = candidates[candidates['nova_group'] == 1]
        if not same_nova_candidates.empty:
            top_recs = list(zip(same_nova_candidates['product_name'], same_nova_candidates['nova_group']))[:topN]
            return [("You have a very least-processed food but here is an alternate:")] + top_recs
    

    same_nova_candidates = candidates[candidates['nova_group'] == product_nova]
    if not same_nova_candidates.empty:
        top_recs = list(zip(same_nova_candidates['product_name'], same_nova_candidates['nova_group']))[:topN]
        return [("Couldn't find less processed options. Try these with same NOVA:", None)] + top_recs
    

    return "Sorry, we couldn't find any suitable alternates. We'll work on improving our recommendations!"
