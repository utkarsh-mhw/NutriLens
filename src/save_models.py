import pickle
import os

def save_unsupervised_models(reducer, kmeans_model):

    save_dir_models = '../models/'


    with open(f'{save_dir_models}umap_reducer.pkl', 'wb') as f:
        pickle.dump(reducer, f)
    print("UMAP reducer saved")
    
    with open(f'{save_dir_models}kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)
    print("KMeans model saved")

def save_supervised_models(model_xgb, tfidf, pca_tfidf, mlb_cat, svd_cats, le):
    save_dir_models = '../models/'

    with open(f'{save_dir_models}xgb_nova_model.pkl', 'wb') as f:
        pickle.dump(model_xgb, f)
    print("XGBoost model saved")
    
    with open(f'{save_dir_models}tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    print("TF-IDF vectorizer saved")
    
    with open(f'{save_dir_models}pca_tfidf.pkl', 'wb') as f:
        pickle.dump(pca_tfidf, f)
    print("PCA (TF-IDF) saved")
    
    with open(f'{save_dir_models}mlb_categories.pkl', 'wb') as f:
        pickle.dump(mlb_cat, f)
    print("MultiLabelBinarizer saved")
    
    with open(f'{save_dir_models}svd_categories.pkl', 'wb') as f:
        pickle.dump(svd_cats, f)
    print("TruncatedSVD saved")
    
    with open(f'{save_dir_models}label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("LabelEncoder saved")

    # with open(f'{save_dir_models}numeric_cols.pkl', 'wb') as f:
    #     pickle.dump(numeric_cols_m, f)
    # print("Numeric columns list saved")



    
