import pandas as pd
from missingDataScript import find_missing_values
from medianImputer import impute_missing_values  

file_path = 'k_titanic.csv'
df = pd.read_csv(file_path)
missing_summary = find_missing_values(df)
imputed_df = impute_missing_values(df, strategy='median')

imputed_file_path = 'imputed_titanic.csv'
imputed_df.to_csv(imputed_file_path, index=False)

print("Imputed data saved to:", imputed_file_path)
