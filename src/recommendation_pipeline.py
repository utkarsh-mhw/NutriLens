import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import umap
import time

from k_means import *

def load_supervised_models():
    models_dir = '../models/'
    with open(f'{models_dir}xgb_nova_model.pkl', 'rb') as f:
        _xgb_model = pickle.load(f)
    
    with open(f'{models_dir}tfidf_vectorizer.pkl', 'rb') as f:
        _tfidf = pickle.load(f)
    
    with open(f'{models_dir}pca_tfidf.pkl', 'rb') as f:
        _pca_tfidf = pickle.load(f)
    
    with open(f'{models_dir}mlb_categories.pkl', 'rb') as f:
        _mlb_cat = pickle.load(f)
    
    with open(f'{models_dir}svd_categories.pkl', 'rb') as f:
        _svd_cats = pickle.load(f)
    
    with open(f'{models_dir}label_encoder.pkl', 'rb') as f:
        _le = pickle.load(f)
    
    # with open(f'{models_dir}numeric_cols.pkl', 'rb') as f:
    #     _numeric_cols = pickle.load(f)
    
    return {
        'xgb': _xgb_model,
        'tfidf': _tfidf,
        'pca_tfidf': _pca_tfidf,
        'mlb_cat': _mlb_cat,
        'svd_cats': _svd_cats,
        'le': _le
    }

def predict_nova_score(product_row):


    models = load_supervised_models()
    numeric_cols_m = ['ingredient_count', 'additives_n', 'additive_density','nutriscore_score', 'monounsaturated-fat_100g', 'sugars_100g',
       'polyunsaturated-fat_100g', 'sugar_fiber_ratio', 'carbohydrates_100g',
       'fiber_100g', 'saturated-fat_100g', 'sodium_per_calorie',
       'allergens_en']
    
    # Extract embeddings
    product_embeddings, _ = extract_embeddings(product_row, start_col=16)
    
    # Extract numeric features
    numeric_vals = product_row[numeric_cols_m].values.astype(np.float32)
    
    # Combine embeddings + numeric
    X_basic = np.hstack([product_embeddings, numeric_vals])
    
    # Process product name with TF-IDF + PCA
    product_name = product_row['product_name'].values[0]
    tfidf_features = models['tfidf'].transform([product_name])
    pca_tfidf_features = models['pca_tfidf'].transform(tfidf_features.toarray())
    
    # Process categories with MLB + SVD
    categories = product_row['categories_tags'].values[0]
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(',')]
    cats_encoded = models['mlb_cat'].transform([categories])
    svd_cats_features = models['svd_cats'].transform(cats_encoded)
    
    # Concatenate all features
    X_final = np.hstack([
        X_basic,
        pca_tfidf_features.astype(np.float32),
        svd_cats_features.astype(np.float32)
    ])
    
    # Predict
    prediction = models['xgb'].predict(X_final)[0]
    
    # Convert back to NOVA scale (1-4)
    nova_score = int(prediction + 1)
    
    return nova_score

def predict_cluster(demo_product):
    start_time = time.time()
    
    # open models and reducer
    models_dir = '../models/'
    with open(f'{models_dir}umap_reducer.pkl', 'rb') as f:
            umap_reducer = pickle.load(f)

    data_dir = '../data/'
    with open(f'{models_dir}kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)

    load_time = time.time()

    product_embeddings, product_meta = extract_embeddings(demo_product)
    embeddings_reduced = umap_reducer.transform(product_embeddings)
    cluster_ids = kmeans_model.predict(embeddings_reduced)

    end_time = time.time()
    print(f"[predict_cluster] Model load time: {load_time - start_time:.4f}s")
    print(f"[predict_cluster] Prediction time: {end_time - load_time:.4f}s")
    print(f"[predict_cluster] Total time: {end_time - start_time:.4f}s")

    return cluster_ids[0]



def recommend_alternates(product_row, product_nova, topN=3, isSameCluster=True):
    
    # product_row = fetch_user_product_details(product_name)
    product_name = product_row['product_name']
    product_cluster_id = predict_cluster(product_row)
    product_row['cluster_id'] = product_cluster_id
    product_row['category_list'] = product_row['categories_tags'].fillna('').apply(lambda x: [c.strip() for c in x.split(',')])

    start_time = time.time()
    df = pd.read_csv("../data/analysis_data/kmeans_output_final.csv")
    df = df.copy()
    time_to_load = time.time()
    df['category_list'] = df['categories_tags'].fillna('').apply(lambda x: [c.strip() for c in x.split(',')])
    time_to_clean = time.time()
    print(f"[loading cluster outputs] Time taken: {time_to_load - start_time:.4f}s")
    print(f"[cleaning categories] Time taken: {time_to_clean - time_to_load:.4f}s")
    


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


    
def fetch_user_product_details(product_name):
    start_time = time.time()

    temp_df = pd.read_csv("../data/analysis_data/demo_data.csv", index_col=0)
    lookup_product = temp_df[temp_df['product_name'] == product_name]
    
    end_time = time.time()
    print(f"[fetch_user_product_details] Time taken: {end_time - start_time:.4f}s")

    # get nova score from predict_nova_scores_here passing lookup_product as argument
    product_nova_score = predict_nova_score(lookup_product)
    # get alternates from recommend_alternates passing nova_score and lookup_product as argument
    alternates = recommend_alternates(lookup_product, product_nova_score)
    # store and return everything in dictionary
    return product_nova_score, alternates