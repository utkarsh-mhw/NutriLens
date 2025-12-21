import pandas as pd
import re



def features_to_keep(df, cols_to_keep = None):

    if cols_to_keep is None:
        cols_to_keep = [
        
        #product metadata
        'code', 
        'product_name',
        'categories_tags',

        # important features based on EDA
        'ingredients_text',
        'ingredient_count',
        'additives_n',
        'additive_density',
        
        #  mid level features based on EDA
        'nutriscore_score',
        'monounsaturated-fat_100g',
        'sugars_100g',
        'polyunsaturated-fat_100g',
        'sugar_fiber_ratio',
        
        #optional
        'carbohydrates_100g',
        'fiber_100g',
        'saturated-fat_100g',
        'sodium_per_calorie',

        # other features that may be useful for additional use cases
        'allergens_en',

        # Target variable
        'nova_group'

    ]
    
    existing_cols = [col for col in cols_to_keep if col in df.columns]

    print(f"Keeping {len(existing_cols)} columns for modeling.")
    df_model = df[existing_cols].copy()
    
    return df_model


def remove_null_ingredients(df):

    before = len(df)
    df_cleaned = df[df['ingredients_text'].notna()].copy()
    after = len(df_cleaned)
    removed = before - after
    

    print(f"Removed {removed:,} rows with null ingredients_text")
    print(f"Remaining: {after:,} rows")

    
    return df_cleaned

def strip_whitespace_ingredients(df):

    df['ingredients_text'] = df['ingredients_text'].str.strip()
    print("Stripped leading and trailing whitespace from ingredients_text")
    return df

def convert_to_lowercase_ingredients(df):

    df['ingredients_text'] = df['ingredients_text'].str.lower()
    print("Converted ingredients_text to lowercase")
    return df

def remove_extra_spaces_ingredients(df):
    # Remove extra spaces
    df['ingredients_text'] = df['ingredients_text'].str.replace(r'\s+', ' ', regex=True)
    print("Removed extra spaces in ingredients_text")
    return df

def remove_special_characters_ingredients(df):
    # Goal: remove standalone numbers (including percent-values) and other special symbols
    # but preserve numbers that are part of vitamin tokens like 'vitamin d3' or 'vitamin b12'

    def _clean_text(s: str) -> str:
        if s is None:
            return s
        # work on lowercase text (pipeline already lowercases earlier)
        text = s

        # find vitamin tokens (allow optional space between letter(s) and digits)
        # examples matched: 'vitamin d3', 'vitamin b12', 'vitamin k 2'
        vit_pattern = re.compile(r"vitamin\s+([a-z]{1,3})\s*(\d{1,3})")
        placeholders = []
        def _placeholder(m):
            idx = len(placeholders)
            token = f"vitamin {m.group(1)}{m.group(2)}"  # normalize to 'vitamin d3'
            placeholders.append(token)
            return f"__VIT_{idx}__"

        # protect vitamin tokens by replacing them with placeholders
        text = vit_pattern.sub(_placeholder, text)

        # remove numbers optionally followed by percent sign (e.g., '100' or '100%')
        text = re.sub(r"\b\d+%?\b", "", text)

        # remove special characters except letters, numbers, whitespace, and dashes (hyphen, en-dash, em-dash)
        text = re.sub(r"[^\w\s\-\u2013\u2014]", " ", text)

        # normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # restore vitamin placeholders back to normalized tokens
        for i, token in enumerate(placeholders):
            text = text.replace(f"__VIT_{i}__", token)

        return text

    df['ingredients_text'] = df['ingredients_text'].apply(_clean_text)
    print("Cleaned ingredients_text: removed non-vitamin numbers and special characters (preserving vitamin tokens)")
    return df

def clean_ingredient_text(df):

    df2 = df.copy()

    df2 = strip_whitespace_ingredients(df2)
    df2 = convert_to_lowercase_ingredients(df2)
    df2 = remove_extra_spaces_ingredients(df2)
    df2 = remove_special_characters_ingredients(df2)
    
    return df2


def fill_na_nutritional_values(df, fill_value=0):

    nut_cols = [col for col in df.columns if col.endswith('_100g')]
    df[nut_cols] = df[nut_cols].fillna(fill_value)
    return df


def stem_and_remove_stopwords_ingredients(df):
    """Stem words in `ingredients_text` and remove English stopwords.

    - Uses NLTK's PorterStemmer and the English stopword list.
    - Attempts to download stopwords if they are missing.
    - Preserves dash characters in tokens.
    """

    # Import NLTK objects here to avoid requiring nltk at module import time
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
    except Exception:
        # try to download stopwords if missing
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    token_re = re.compile(r"[A-Za-z\u2013\u2014-]+")

    def _process(text: str) -> str:
        if text is None:
            return text
        tokens = token_re.findall(text)
        out_tokens = []
        for tok in tokens:
            tl = tok.lower()
            if tl in stop_words:
                continue
            stem = stemmer.stem(tl)
            out_tokens.append(stem)
        return " ".join(out_tokens)

    df['ingredients_text'] = df['ingredients_text'].apply(_process)
    print("Stemmed ingredients_text and removed stopwords")
    return df

from langdetect import detect, LangDetectException

def keep_english_ingredients(df):
    def is_english(text):
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return False
    mask = df['ingredients_text'].apply(is_english)
    filtered_df = df[mask].copy()
    print(f"Kept {filtered_df.shape[0]} English rows out of {df.shape[0]}")
    return filtered_df

def filter_if_all_present(cluster_df, allowed_categories):
    allowed_set = set(allowed_categories)

    def check_row(x):
        if pd.isna(x):
            return False
        cats = str(x).split(',')
        return all(cat.strip() in allowed_set for cat in cats)

    mask = cluster_df['categories_tags'].apply(check_row)
    return cluster_df[mask].copy()

def filter_if_one_present(cluster_df, filt_categories):
    allowed_categories = ["en:baby-foods", "en:fishes-and-their-products", "en:chocolates", "en:salads", "en:ice-creams-and-sorbets", "en:pastas", "en:bodybuilding-supplements", "en:canned-soups", "en:cheeses", "en:sodas"]
    allowed_set = set(allowed_categories)

    def check_row(x):
        if pd.isna(x):
            return False
        cats = str(x).split(',')
        return any(cat.strip() in allowed_set for cat in cats)

    mask = cluster_df['categories_tags'].apply(check_row)
    return cluster_df[mask].copy()