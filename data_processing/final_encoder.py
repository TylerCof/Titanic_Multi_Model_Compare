#final encoder


import pandas as pd
from one_hot_encoder_final_script import apply_one_hot_encoding

'''
# Example usage:
file_path = 'covtype.csv'
df = pd.read_csv(file_path)
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade']
encoded_df = apply_one_hot_encoding(df, categorical_columns)
print(encoded_df.head())
'''

file_path = 'cleaned_and_imputed_data.csv'
df = pd.read_csv(file_path)
categorical_columns = ['Embarked', 'Sex']
encoded_df = apply_one_hot_encoding(df,categorical_columns)

# Save the cleaned and imputed DataFrame to a new CSV file
encoded_df.to_csv('encoded_cleaned_and_imputed_data.csv', index=False)
