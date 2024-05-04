import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def apply_one_hot_encoding(data, categorical_columns):
    """
    Apply one-hot encoding to specified categorical columns in a DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - categorical_columns (list): List of column names to be one-hot encoded.

    Returns:
    - pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    # Create a OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse_output=False)

    # Apply one-hot encoding to specified columns
    categorical_encoded = one_hot_encoder.fit_transform(data[categorical_columns])

    # Convert the array returned by OneHotEncoder to a DataFrame
    categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns))

    # Drop original categorical columns and concatenate the encoded DataFrame
    data_encoded = pd.concat([data.drop(columns=categorical_columns).reset_index(drop=True), categorical_encoded_df], axis=1)

    return data_encoded
'''
# Example usage:
file_path = 'covtype.csv'
df = pd.read_csv(file_path)
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade']
encoded_df = apply_one_hot_encoding(df, categorical_columns)
print(encoded_df.head())
'''





