import pandas as pd

input_path = "cleaned_headlines.csv"
output_path = "final_headlines.csv"

df = pd.read_csv(input_path)

df_clean = df[df["headline"].notnull() & (df["headline"].str.strip() != "")]

df_clean.to_csv(output_path, index=False)
