import pandas as pd
from sklearn.impute import SimpleImputer

def impute_missing_values(data, threshold=0.5,strategy='median'):
    """
    Remove columns with a proportion of missing values above a certain threshold,
    and then impute missing values in the 'age' column using SimpleImputer.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - threshold (float): The maximum proportion of missing values allowed for a column to be retained.
    - strategy (str): Imputation strategy ('mean', 'median', 'most_frequent', or 'constant').

    Returns:
    - pd.DataFrame: The DataFrame with columns containing more missing values than the threshold removed
      and the 'age' column imputed with the specified strategy.
    """
    # Remove columns with more missing values than the threshold
    cleaned_data = remove_columns_with_missing_values(data, threshold)

    # Impute missing values in the 'age' column with the specified strategy
    age_column = 'Age'
    if age_column in cleaned_data.columns:
        imputer = SimpleImputer(strategy=strategy)
        cleaned_data[age_column] = imputer.fit_transform(cleaned_data[[age_column]])
     
    return cleaned_data

# Function to remove columns with missing values exceeding a threshold
def remove_columns_with_missing_values(data, threshold=0.5):
    """
    Remove columns with a proportion of missing values above a certain threshold.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - threshold (float): The maximum proportion of missing values allowed for a column to be retained.

    Returns:
    - pd.DataFrame: The DataFrame with columns containing more missing values than the threshold removed.
    """
    # Calculate the proportion of missing values in each column
    missing_ratios = data.isnull().mean()

    # Identify columns with missing values exceeding the threshold
    columns_to_drop = missing_ratios[missing_ratios > threshold].index

    # Drop the identified columns
    cleaned_data = data.drop(columns=columns_to_drop)

    return cleaned_data

# Example usage:
file_path = 'k_titanic.csv'
df = pd.read_csv(file_path)

# Remove columns with more than 50% missing values and impute 'age' column with median
cleaned_and_imputed_df = impute_missing_values(df, threshold=0.5,strategy='median')
print(cleaned_and_imputed_df.head())

# Save the cleaned and imputed DataFrame to a new CSV file
cleaned_and_imputed_df.to_csv('cleaned_and_imputed_data.csv', index=False)
