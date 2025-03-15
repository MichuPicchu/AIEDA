import numpy as np

def get_table_shape(df):
    """Get the shape of the DataFrame"""
    return {
        'rows': df.shape[0],
        'columns': df.shape[1]
    }

def get_duplicate_and_na_counts(df):
    """Get counts of duplicate rows and rows with NA values"""
    duplicate_count = df.duplicated().sum()
    na_count = df.isna().sum().sum()
    return {
        'duplicate_count': duplicate_count,
        'na_count': na_count
    }

def get_column_datatypes(df):
    """Get data types for each column"""
    return {
        'data_types': {
            col: str(dtype) for col, dtype in df.dtypes.items()
        }
    }

def get_missing_values(df):
    """Count missing values per column"""
    return {
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum()
    }

def get_unique_values(df, max_unique=10):
    """Get unique values for categorical columns"""
    unique_vals = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            uniques = df[col].unique().tolist()
            unique_vals[col] = {
                'count': len(uniques),
                'values': uniques[:max_unique]
            }
    return {'unique_values': unique_vals}

def get_numeric_stats(df):
    """Get statistics for numeric columns"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    stats = {}
    
    for col in numeric_cols:
        stats[col] = {
            'min': round(df[col].min(),3),
            'max': round(df[col].max(),3),
            'mean': round(df[col].mean(),3),
            'median': round(df[col].median(),3),
            'std': round(df[col].std(),3),
            'skew': round(df[col].skew(),3)
        }
    return {'numeric_statistics': stats}

def detect_outliers(df, threshold=1.5):
    """Detect outliers using IQR method"""
    outliers = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = round(q1 - (threshold * iqr),2)
        upper_bound = round(q3 + (threshold * iqr),2)
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers[col] = {
            'count': outlier_mask.sum(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'num_outlier_values': len(df[col][outlier_mask]),
        }
    
    return {'outliers': outliers}

def get_example_rows(df, n=2):
    """Get example rows from the dataframe"""
    return {
        'example_rows': df.head(n).to_dict(orient='records')
    }
