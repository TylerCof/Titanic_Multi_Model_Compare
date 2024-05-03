import pandas as pd

def find_missing_values(data):
    """
    Find missing values in a DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.

    Returns:
    - dict: A dictionary containing information about missing values.
    """
    missing_data = data.isnull()
    missing_columns = missing_data.any()
    missing_rows = missing_data.any(axis=1)
    total_missing = missing_data.sum().sum()
    missing_per_column = data.isnull().sum()
    missing_locations = [(index, column) for index, row in missing_data.iterrows() for column in missing_data.columns if row[column]]

    missing_summary = {
        "total_missing": total_missing,
        "missing_per_column": missing_per_column,
        "missing_locations": missing_locations[:10]  # Display first 10 locations for brevity
    }

    return missing_summary
'''
# Example usage:
file_path = 'covtype.csv'
df = pd.read_csv(file_path)
missing_summary = find_missing_values(df)
print(missing_summary)
'''

