import pickle
import os


# def save_unsupervised_models(umap_reducer, kmeans_model):
def save_unsupervised_models(reducer, kmeans_model):

    save_dir_models = '../models/'


    with open(f'{save_dir_models}umap_reducer.pkl', 'wb') as f:
        pickle.dump(reducer, f)
    print("UMAP reducer saved")
    
    with open(f'{save_dir_models}kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)
    print("KMeans model saved")


