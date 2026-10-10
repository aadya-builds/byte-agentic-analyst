import pandas as pd

df = pd.read_csv("netflix_titles.csv")
print(df["country"].dropna().str.split(", ").explode().str.strip().value_counts().head(3))
print(df["rating"].value_counts().head(3))
print(df.groupby("type")["release_year"].mean().round(2))
q1, q3 = df["release_year"].quantile(0.25), df["release_year"].quantile(0.75)
iqr = q3 - q1
print(len(df[(df["release_year"] < q1 - 1.5*iqr) | (df["release_year"] > q3 + 1.5*iqr)]))