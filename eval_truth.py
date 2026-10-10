import pandas as pd

df = pd.read_csv("train.csv")

print("1. Rows:", len(df))
print("2. Cabin missing %:", round(df["Cabin"].isna().mean() * 100, 1))
print("3. Average Age:", round(df["Age"].mean(), 2))
print("4. Average Fare by Sex:\n", df.groupby("Sex")["Fare"].mean().round(2))
print("5. Survival rate by Sex:\n", df.groupby("Sex")["Survived"].mean().round(3))
print("6. Survival rate by Pclass:\n", df.groupby("Pclass")["Survived"].mean().round(3))

q1, q3 = df["Fare"].quantile(0.25), df["Fare"].quantile(0.75)
iqr = q3 - q1
outliers = df[(df["Fare"] < q1 - 1.5 * iqr) | (df["Fare"] > q3 + 1.5 * iqr)]
print("7. Fare outliers:", len(outliers))

print("8. Correlation with Survived:\n", df.corr(numeric_only=True)["Survived"].round(2))
print("9. Embarked counts:\n", df["Embarked"].value_counts())
print("10. Median Fare:", df["Fare"].median())