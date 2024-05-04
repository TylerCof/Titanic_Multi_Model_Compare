#This will remove and data columns that are unnecessary or harmful for the model (Name, Passanger ID, Ticke)

import pandas as pd

'''
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



'''


file_path = 'encoded_cleaned_and_imputed_data.csv'
df = pd.read_csv(file_path)
columns_to_drop = ['Name','Ticket','PassengerId']
final_data = df.drop(columns=columns_to_drop)
final_data.to_csv('final_data.csv', index=False)


