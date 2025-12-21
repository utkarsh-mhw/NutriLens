import pandas as pd
import matplotlib.pyplot as plt

def check_nova_distribution(df):

    print("="*50)
    print("NOVA GROUP DISTRIBUTION")
    print("="*50)
    

    null_count = df['nova_group'].isnull().sum()
    print(f"Null NOVA values: {null_count}")
    

    print("\nNOVA Group Counts:")
    print(df['nova_group'].value_counts().sort_index())

    

    fig, ax = plt.subplots(figsize=(8, 5))
    

    df['nova_group'].value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('NOVA Group Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('NOVA Group')
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)
    plt.show()

    return df['nova_group'].value_counts()


def analyze_missing_data(df, top_n=20):
    print("="*50)
    print("MISSING DATA ANALYSIS")
    print("="*50)
    

    cols_to_analyze = [col for col in df.columns if not col.endswith('_100g')]
    
    missing_stats = pd.DataFrame({
        'column': cols_to_analyze,
        'missing_count': df[cols_to_analyze].isnull().sum().values,
        'missing_pct': (df[cols_to_analyze].isnull().sum().values / len(df) * 100)
    }).sort_values('missing_pct', ascending=False)
    
    print(f"\nTop {top_n} columns with missing data:")
    print(missing_stats.head(top_n).to_string(index=False))
    
    top_missing = missing_stats.head(top_n)
    plt.figure(figsize=(12, 6))
    plt.barh(top_missing['column'], top_missing['missing_pct'], color='coral')
    plt.xlabel('Missing Percentage (%)')
    plt.title(f'Top {top_n} Columns with Missing Data', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return missing_stats

def analyze_ingredient_text(df):

    print("="*50)
    print("INGREDIENTS TEXT ANALYSIS")
    print("="*50)
    
    # Missing values
    missing = df['ingredients_text'].isnull().sum()
    print(f"Missing ingredients_text: {missing:,} ({missing/len(df)*100:.2f}%)")
    
    # Basic stats
    df['temp_ingredient_length'] = df['ingredients_text'].fillna('').apply(len)
    df['temp_ingredient_count'] = df['ingredients_text'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    
    print(f"\nIngredient text length stats:")
    print(df['temp_ingredient_length'].describe())
    
    print(f"\nIngredient count stats:")
    print(df['temp_ingredient_count'].describe())
    
    # By NOVA group
    print(f"\nApproximate ingredient count by NOVA group:")
    print(df.groupby('nova_group')['temp_ingredient_count'].describe())
    
    # Cleanup temp columns
    df.drop(['temp_ingredient_length', 'temp_ingredient_count'], axis=1, inplace=True)
    
    return missing


def correlation_with_nova(df, columns_to_check=None):

    print("="*50)
    print("CORRELATION WITH NOVA GROUP")
    print("="*50)
    
    # Default columns if none provided
    if columns_to_check is None:
        columns_to_check = [
            # Core macronutrients
            'energy-kcal_100g', 
            'fat_100g', 
            'saturated-fat_100g', 
            'trans-fat_100g',
            'carbohydrates_100g', 
            'sugars_100g', 
            'added-sugars_100g',
            'fiber_100g', 
            'proteins_100g', 
            'salt_100g', 
            'sodium_100g',
            
            # Fat breakdown
            'monounsaturated-fat_100g',
            'polyunsaturated-fat_100g',
            'omega-3-fat_100g',
            'omega-6-fat_100g',
            
            # Sugar types
            'sucrose_100g',
            'glucose_100g',
            'fructose_100g',
            'lactose_100g',
            'maltose_100g',
            'maltodextrins_100g',
            
            # Sugar alcohols
            'polyols_100g',
            'erythritol_100g',
            'maltitol_100g',
            'sorbitol_100g',
            
            # Additives (critical!)
            'additives_n',
            
            # Scores
            'nutriscore_score',
            'nutrition-score-fr_100g',
        ]
    
    # Create derived features if they don't exist
    derived_features = {}
    
    # Ingredient count
    if 'ingredient_count' not in df.columns and 'ingredients_text' in df.columns:
        df['ingredient_count'] = df['ingredients_text'].fillna('').apply(
            lambda x: len(x.split(',')) if x else 0
        )
        derived_features['ingredient_count'] = 'Created'
        columns_to_check.append('ingredient_count')
    elif 'ingredient_count' in df.columns:
        if 'ingredient_count' not in columns_to_check:
            columns_to_check.append('ingredient_count')
    
    # Additive density (additives per ingredient)
    if 'additives_n' in df.columns and 'ingredient_count' in df.columns:
        if 'additive_density' not in df.columns:
            df['additive_density'] = df['additives_n'] / (df['ingredient_count'] + 1)
            derived_features['additive_density'] = 'Created'
            columns_to_check.append('additive_density')
    
    # Sugar to fiber ratio
    if 'sugars_100g' in df.columns and 'fiber_100g' in df.columns:
        if 'sugar_fiber_ratio' not in df.columns:
            df['sugar_fiber_ratio'] = df['sugars_100g'] / (df['fiber_100g'] + 0.1)
            derived_features['sugar_fiber_ratio'] = 'Created'
            columns_to_check.append('sugar_fiber_ratio')
    
    # Sodium density
    if 'sodium_100g' in df.columns and 'energy-kcal_100g' in df.columns:
        if 'sodium_per_calorie' not in df.columns:
            df['sodium_per_calorie'] = df['sodium_100g'] / (df['energy-kcal_100g'] + 1)
            derived_features['sodium_per_calorie'] = 'Created'
            columns_to_check.append('sodium_per_calorie')
    
    # Saturated fat percentage
    if 'saturated-fat_100g' in df.columns and 'fat_100g' in df.columns:
        if 'saturated_fat_pct' not in df.columns:
            df['saturated_fat_pct'] = df['saturated-fat_100g'] / (df['fat_100g'] + 0.1)
            derived_features['saturated_fat_pct'] = 'Created'
            columns_to_check.append('saturated_fat_pct')
    
    if derived_features:
        print("\n✓ Created derived features:")
        for feat, status in derived_features.items():
            print(f"  - {feat}")
    
    # Filter to existing columns
    existing_cols = [col for col in columns_to_check if col in df.columns]
    
    # Calculate correlations
    correlations = []
    for col in existing_cols:
        non_null_count = df[col].notna().sum()
        if non_null_count > 100:
            corr = df[col].corr(df['nova_group'])
            if pd.notna(corr):
                correlations.append({
                    'feature': col, 
                    'correlation': corr,
                    'non_null_count': non_null_count,
                    'missing_pct': (1 - non_null_count/len(df)) * 100
                })
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
    
    # Print results
    print("-"*70)
    print("Top 20 Correlations with NOVA group:")
    print("-"*70)
    print(corr_df[['feature', 'correlation', 'missing_pct']].head(20).to_string(index=False))
    
    # print("-"*70)
    # print("Strongest Positive Correlations (higher value = higher NOVA):")
    # print("-"*70)
    # top_positive = corr_df[corr_df['correlation'] > 0].head(10)
    # print(top_positive[['feature', 'correlation']].to_string(index=False))
    
    # print("-"*70)
    # print("Strongest Negative Correlations (higher value = lower NOVA):")
    # print("-"*70)
    # top_negative = corr_df[corr_df['correlation'] < 0].head(10)
    # print(top_negative[['feature', 'correlation']].to_string(index=False))
    
    # Visualize top correlations
    top_n = min(25, len(corr_df))
    top_corr = corr_df.head(top_n)
    
    # plt.figure(figsize=(12, 8))
    # colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_corr['correlation']]
    # bars = plt.barh(top_corr['feature'], top_corr['correlation'], color=colors)
    
    # plt.xlabel('Correlation with NOVA Group', fontsize=12)
    # plt.ylabel('Feature', fontsize=12)
    # plt.title(f'Feature Correlation with NOVA Group', fontsize=14, fontweight='bold')
    # plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    # plt.grid(axis='x', alpha=0.3)
    
    # # Add correlation values on bars
    # for i, (bar, corr) in enumerate(zip(bars, top_corr['correlation'])):
    #     plt.text(corr + 0.01 if corr > 0 else corr - 0.01, i, f'{corr:.3f}', 
    #             va='center', ha='left' if corr > 0 else 'right', fontsize=9)
    
    # plt.tight_layout()
    # plt.savefig("feature_correlation.png", dpi=300, bbox_inches='tight')
    # plt.show()
    

    plt.figure(figsize=(12, 5))   # Reduced height (was 12x8)

    # Colors based on correlation sign
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_corr['correlation']]

    # Thinner bars using height parameter
    bars = plt.barh(
        top_corr['feature'], 
        top_corr['correlation'], 
        color=colors,
        height=0.55              # reduced bar thickness (default 0.8)
    )

    plt.xlabel('Correlation with NOVA Group', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Correlation with NOVA Group', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(axis='x', alpha=0.3)

    # Reduce spacing (padding) between bars
    plt.margins(y=0.02)

    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, top_corr['correlation'])):
        plt.text(
            corr + 0.01 if corr > 0 else corr - 0.01,
            i,
            f'{corr:.3f}',
            va='center',
            ha='left' if corr > 0 else 'right',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig("feature_correlation.png", dpi=300, bbox_inches='tight')
    plt.show()
    # Summary statistics
    print("="*50)
    print("Summary")
    print("="*50)
    print(f"Total features analyzed: {len(corr_df)}")
    print(f"Features with |correlation| > 0.1: {(corr_df['correlation'].abs() > 0.1).sum()}")
    print(f"Features with |correlation| > 0.2: {(corr_df['correlation'].abs() > 0.2).sum()}")
    print(f"Features with |correlation| > 0.3: {(corr_df['correlation'].abs() > 0.3).sum()}")
    
    if len(corr_df) > 0:
        print(f"\nStrongest correlation: {corr_df.iloc[0]['feature']} ({corr_df.iloc[0]['correlation']:.3f})")
        print(f"Weakest correlation: {corr_df.iloc[-1]['feature']} ({corr_df.iloc[-1]['correlation']:.3f})")
    
    return corr_df




def run_eda_pipeline(df):

    

    nova_dist = check_nova_distribution(df)
    

    missing_stats = analyze_missing_data(df, top_n=25)
    

    missing_ingredients = analyze_ingredient_text(df)
    

    # correlations = correlation_with_nova(df)
    

    
    return {
        'nova_distribution': nova_dist,
        'missing_stats': missing_stats
        # 'correlations': correlations
    }


