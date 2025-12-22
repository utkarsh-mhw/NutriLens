import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import umap
import time

from src.k_means import extract_embeddings


_models_loaded = False
_supervised_models = None
_unsupervised_models = None
_demo_data = None
_recommendation_db = None

def _load_all_models():

    global _models_loaded, _supervised_models, _unsupervised_models, _demo_data, _recommendation_db
    
    if not _models_loaded:
        
        models_dir = 'models/'
        
        with open(f'{models_dir}xgb_nova_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open(f'{models_dir}tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open(f'{models_dir}pca_tfidf.pkl', 'rb') as f:
            pca_tfidf = pickle.load(f)
        with open(f'{models_dir}mlb_categories.pkl', 'rb') as f:
            mlb_cat = pickle.load(f)
        with open(f'{models_dir}svd_categories.pkl', 'rb') as f:
            svd_cats = pickle.load(f)
        with open(f'{models_dir}label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        
        _supervised_models = {
            'xgb': xgb_model,
            'tfidf': tfidf,
            'pca_tfidf': pca_tfidf,
            'mlb_cat': mlb_cat,
            'svd_cats': svd_cats,
            'le': le
        }
        with open(f'{models_dir}umap_reducer.pkl', 'rb') as f:
            umap_reducer = pickle.load(f)
        with open(f'{models_dir}kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        
        _unsupervised_models = {
            'umap': umap_reducer,
            'kmeans': kmeans_model
        }
        
        _demo_data = pd.read_csv("data/analysis_data/demo_data.csv", index_col=0)
        
        _recommendation_db = pd.read_csv("data/analysis_data/kmeans_output_final.csv")
        _recommendation_db['category_list'] = _recommendation_db['categories_tags'].fillna('').apply(
            lambda x: [c.strip() for c in x.split(',')] if isinstance(x, str) else []
        )
        
        _models_loaded = True
    
    return _supervised_models, _unsupervised_models, _demo_data, _recommendation_db


def predict_nova_and_cluster(product_row):

    supervised, unsupervised, _, _ = _load_all_models()
    
    numeric_cols_m = [
        'ingredient_count', 'additives_n', 'additive_density', 'nutriscore_score',
        'monounsaturated-fat_100g', 'sugars_100g', 'polyunsaturated-fat_100g',
        'sugar_fiber_ratio', 'carbohydrates_100g', 'fiber_100g',
        'saturated-fat_100g', 'sodium_per_calorie', 'allergens_en'
    ]
    
    product_embeddings, _ = extract_embeddings(product_row)
    numeric_vals = product_row[numeric_cols_m].values.astype(np.float32)
    
    X_basic = np.hstack([product_embeddings, numeric_vals])
    

    product_name = product_row['product_name'].values[0]
    tfidf_features = supervised['tfidf'].transform([product_name])
    pca_tfidf_features = supervised['pca_tfidf'].transform(tfidf_features.toarray())

    

    categories = product_row['categories_tags'].values[0]
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(',')]
    cats_encoded = supervised['mlb_cat'].transform([categories])
    svd_cats_features = supervised['svd_cats'].transform(cats_encoded)

    

    X_final = np.hstack([
        X_basic,
        pca_tfidf_features.astype(np.float32),
        svd_cats_features.astype(np.float32)
    ])
    

    nova_prediction = supervised['xgb'].predict(X_final)[0]
    nova_score = int(nova_prediction + 1)
    

    embeddings_reduced = unsupervised['umap'].transform(product_embeddings)
    cluster_id = unsupervised['kmeans'].predict(embeddings_reduced)[0]
    
    return nova_score, int(cluster_id), categories

def recommend_alternates(product_name, product_nova, product_cluster, user_categories, topN=3, isSameCluster=True):

    _, _, _, df = _load_all_models()
    

    candidates = df[df['product_name'] != product_name].copy()
    
    if isSameCluster:
        candidates = candidates[candidates['cluster_id'] == product_cluster]
    

    candidates = candidates[candidates['category_list'].apply(
        lambda cats: len(set(user_categories).intersection(cats)) > 0
    )]
    

    healthier_candidates = candidates[candidates['nova_group'] < product_nova]
    
    if not healthier_candidates.empty:
        healthier_candidates = healthier_candidates.sort_values('nova_group')
        return list(zip(
            healthier_candidates['product_name'],
            healthier_candidates['nova_group'].astype(int)
        ))[:topN]
    
    # If product is already NOVA 1
    if product_nova == 1:
        same_nova_candidates = candidates[candidates['nova_group'] == 1]
        if not same_nova_candidates.empty:
            recs = list(zip(
                same_nova_candidates['product_name'],
                same_nova_candidates['nova_group'].astype(int)
            ))[:topN]
            return [("You have a very least-processed food but here are alternates:", None)] + recs
    
    # Find same NOVA level alternates
    same_nova_candidates = candidates[candidates['nova_group'] == product_nova]
    if not same_nova_candidates.empty:
        recs = list(zip(
            same_nova_candidates['product_name'],
            same_nova_candidates['nova_group'].astype(int)
        ))[:topN]
        return [("Couldn't find less processed options. Try these with same NOVA:", None)] + recs
    
    # No recommendations found
    return []


def analyze_product(product_name):

    start_time = time.time()
    _, _, demo_data, _ = _load_all_models()
    lookup_product = demo_data[demo_data['product_name'] == product_name]
    
    if lookup_product.empty:
        return {
            'success': False,
            'error': f"Product '{product_name}' not found in database",
            'recommendations': []
        }
    
    nova_score, cluster_id, categories = predict_nova_and_cluster(lookup_product)
    

    recommendations = recommend_alternates(
        product_name=product_name,
        product_nova=nova_score,
        product_cluster=cluster_id,
        user_categories=categories,
        topN=3,
        isSameCluster=True
    )
    
    result = {
        'success': True,
        'product_name': product_name,
        'nova_score': nova_score,
        'cluster_id': cluster_id,
        'recommendations': recommendations if recommendations else [],
        'processing_time': f"{time.time() - start_time:.3f}s"
    }
    
    if not recommendations:
        result['message'] = "Sorry, we couldn't find any suitable alternates at this time."
    
    return result



def fetch_user_product_details(product_name):
    result = analyze_product(product_name)
    
    if result['success']:
        return result['nova_score'], result['recommendations']
    else:
        return None, result.get('error', 'Product not found')



if __name__ == "__main__":
    # Test the system
    test_product = "Tortellini rosa"
    
    
    result = analyze_product(test_product)
    
    if result['success']:
        print(f"Product: {result['product_name']}")
        print(f"NOVA Score: {result['nova_score']}")
        print(f"Cluster: {result['cluster_id']}")
        print(f"Processing Time: {result['processing_time']}")
        
        print(f"Recommendations:")
        if result['recommendations']:
            for rec in result['recommendations']:
                if isinstance(rec, tuple) and rec[1] is not None:
                    print(f"  • {rec[0]} (NOVA: {rec[1]})")
                elif isinstance(rec, tuple):
                    print(f"  {rec[0]}")
        else:
            print(f"{result.get('message', 'No recommendations')}")
    else:
        print(f"Error: {result['error']}")
    

    
    # second call

    result2 = analyze_product(test_product)
    print(f"Second call time: {result2['processing_time']}")