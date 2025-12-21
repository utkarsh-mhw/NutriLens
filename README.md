# ML-project
Project for ML 7641
 
 `/notebooks/`: Contains Jupyter notebooks for data analysis, data cleaning, supervised learning, and unsupervised learning experiments.

 `/notebooks/analysis.ipynb`: Jupyter notebook for data analysis and cleaning. Main functions include:
- Defining and selecting relevant columns from the raw dataset for analysis.
- Performing exploratory data analysis (EDA), including NOVA group label distribution, missing data statistics, and correlation analysis with the NOVA group.
- Data cleaning and feature engineering, such as removing rows whose ingredients_text is null, stripping heading and trailing whitespace, changing all characters to lowercase, removing extra space and special characters, filling missing nutritional values, stemming, stopword removal, andremoving non-English ingredient_text.
- Inspecting, comparing, and saving intermediate and final processed data.
- Clustering analysis using product embeddings and k-means, including merging data, running clustering pipelines, and saving clustered results.
 
 `/notebooks/Supervised_NOVA_Score.ipynb`: Jupyter notebook for supervised learning and NOVA score prediction. Main functions include:
- Loading and preprocessing the dataset, including handling categorical and numeric features, and loading precomputed embeddings.
- Exploratory data analysis (EDA) with correlation heatmaps and unique value inspection.
- PCA dimensionality reduction after TF-IDF vectorization of product names and multilabel binarization of category tags.
- Concatenating numeric, embedding, and engineered features for model input.
- Training and evaluating supervised models (Random Forest and XGBoost) for NOVA group classification, including model fitting, prediction, and classification report generation.
- Analyzing and displaying feature importances for model interpretability.
- Inspecting, transforming, and visualizing data and model results.

 `/notebooks/ml_unsupervised.ipynb`: Jupyter notebook for unsupervised machine learning experiments. Main functions include:
- Preprocessing data: extracting and engineering features such as additive and allergen tags, sweetener flags, and handling missing values.
- Building feature sets for clustering, including numeric, flag, and text features (with TF-IDF and SVD dimensionality reduction).
- Running clustering algorithms (KMeans and Gaussian Mixture Models) with multiple feature sets and cluster counts, and evaluating results using silhouette, Calinski–Harabasz, and Davies–Bouldin scores.
- Profiling clusters by computing medians, means, and tag summaries for numeric features, and extracting top TF-IDF terms for each cluster.
- Finding similar items within clusters using nearest neighbors.
- Saving cluster assignments, summaries, and top terms, and displaying cluster overviews and representative similar items.
 
 `/src/`: Source code for data processing, exploratory data analysis, and unsupervised leaning algorithms.
 /src/data_cleaning.py: Utility functions for cleaning and preprocessing raw data, which are used in  /notebooks/analysis.ipynb. The cleaning techniques include:
 - Removing rows whose ingredients_text is null
 - Stripping heading and trailing whitespace
 - Changing all characters to lowercase
 - Removing extra space and special characters
 - Filling missing nutritional values
 - Stemming, stopword removal
 - Removing non-English ingredient_text
 
 `/src/data_loading.py`: Utility functions for loading writing, and merging dataset, which are used in  /notebooks/analysis.ipynb

 `/src/eda.py`: Utility functions for exploratory data analysis (EDA), which are used in  /notebooks/analysis.ipynb. The EDA functions include:
- Analyzing NOVA group label distribution
- Analyzing missing data statistics
- Analyzing correlation with the NOVA group
 /src/k_means.py: Implementation of the k-means clustering algorithm on embeddings, which is used in /notebooks/analysis.ipynb.
 
 `README.md`: Project overview and directory/file descriptions.
