import pandas as pd
import os


def save_sampled_subset(df, columns=None, nrows=None, out_fname=None):
    """Save a CSV with only the requested columns and up to nrows rows.

    Parameters
    - df: pandas.DataFrame - source dataframe
    - columns: list[str] - columns to keep (order preserved)
    - nrows: int - maximum number of rows to include
    - out_fname: str|None - filename to write under data/analysis_data; when None uses 'subset.csv'

    The file will be written to data/analysis_data/<out_fname> relative to the repo root.
    The directory is created if it doesn't exist.
    """

    # validate inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # if columns not provided, default to all dataframe columns
    if columns is None:
        columns = list(df.columns)

    if not isinstance(columns, (list, tuple)):
        raise TypeError("columns must be a list or tuple of column names")

    # if nrows is None, keep all rows; otherwise validate integer
    if nrows is None:
        nrows = len(df)
    elif not isinstance(nrows, int) or nrows < 0:
        raise ValueError("nrows must be a non-negative integer")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'analysis_data')
    # __file__ is src/data_loading.py -> go up one level to repo root/src then ../data
    # but using dirname(dirname(__file__)) gives repo/src parent = repo
    os.makedirs(out_dir, exist_ok=True)

    out_name = out_fname or 'subset.csv'
    out_path = os.path.join(out_dir, out_name)

    # select columns that exist in df; preserve order
    cols_to_write = [c for c in columns if c in df.columns]

    if len(cols_to_write) == 0:
        raise ValueError("None of the requested columns are present in the dataframe")

    subset = df.loc[:, cols_to_write].head(nrows)
    subset.to_csv(out_path, index=False)

    print(f"Wrote {len(subset):,} rows x {len(cols_to_write)} cols to {out_path}")



def read_file(path_to_file, columns_to_keep=None, seperator = '\t'):

    if columns_to_keep is None:
        df = pd.read_csv(path_to_file, sep=seperator, low_memory=False)
    else:
        df = pd.read_csv(path_to_file, sep=seperator, usecols=columns_to_keep, low_memory=False)
    print(f"Shape of the data: {df.shape}")
    print(f"\nColumns loaded:")
    print(df.columns.tolist())
    return df


    

def read_columns(path_to_file, seperator='\t'):
    df = pd.read_csv(path_to_file, sep=seperator, nrows=0, low_memory=False)
    return df.columns.tolist()


def merge_data_frames(df1, cols1=None, df2=None, cols2=None):
    if not isinstance(df1, pd.DataFrame):
        raise TypeError("df1 must be a pandas DataFrame")
    if df2 is None or not isinstance(df2, pd.DataFrame):
        raise TypeError("df2 must be a pandas DataFrame")

    if cols1 is None:
        cols1 = list(df1.columns)
    if cols2 is None:
        cols2 = list(df2.columns)

    if not isinstance(cols1, (list, tuple)) or not isinstance(cols2, (list, tuple)):
        raise TypeError("cols1 and cols2 must be list or tuple of column names or None")

    # select only existing columns
    sel1 = [c for c in cols1 if c in df1.columns]
    sel2 = [c for c in cols2 if c in df2.columns]

    if len(sel1) == 0 and len(sel2) == 0:
        raise ValueError("None of the requested columns are present in either dataframe")

    df1_sel = df1.loc[:, sel1] if len(sel1) > 0 else df1.iloc[:, 0:0]
    df2_sel = df2.loc[:, sel2] if len(sel2) > 0 else df2.iloc[:, 0:0]

    # Reset index to ensure rows align properly
    df1_sel = df1_sel.reset_index(drop=True)
    df2_sel = df2_sel.reset_index(drop=True)

    # Concatenate horizontally
    result = pd.concat([df1_sel, df2_sel], axis=1)

    return result


