import pandas as pd
from missingDataScript import find_missing_values

file_path = 'cleaned_and_imputed_data.csv'
df = pd.read_csv(file_path)
missing_summary = find_missing_values(df)
print(missing_summary)

