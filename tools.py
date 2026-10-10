import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save charts to a file instead of opening a window
import matplotlib.pyplot as plt
from scipy import stats

df = None


def set_dataframe(new_df):
    """Called by the app to give the tools the dataset the user uploaded."""
    global df
    df = new_df


def _check_columns(*cols):
    """Helper: returns an error message if any column is missing from the dataset."""
    for c in cols:
        if c and c not in df.columns:
            return f"Column '{c}' not found. Available columns: {list(df.columns)}"
    return None


# ---------- TOOL 1 ----------
def get_schema() -> str:
    """Returns every column name, its data type and how many values are missing."""
    lines = [f"{c}: type={df[c].dtype}, missing={int(df[c].isna().sum())}" for c in df.columns]
    return f"Rows: {len(df)}\n" + "\n".join(lines)


# ---------- TOOL 2 ----------
def data_quality_audit() -> str:
    """Audits the dataset for quality problems: duplicate rows, columns with missing values,
    columns with only one unique value, and numeric columns containing negative values."""
    issues = [f"Duplicate rows: {int(df.duplicated().sum())}"]
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            issues.append(f"{col}: {pct:.1f}% values missing")
        if df[col].nunique() <= 1:
            issues.append(f"{col}: has only one unique value, so it is useless")
    for col in df.select_dtypes(include="number").columns:
        neg = int((df[col] < 0).sum())
        if neg > 0:
            issues.append(f"{col}: {neg} negative values")
    return "\n".join(issues)


# ---------- TOOL 3 ----------
def compare_groups(group_column: str, value_column: str) -> str:
    """Compares a numeric column across the groups of a categorical column (for example
    average Fare for each Sex). Runs a statistical test and says whether the difference is
    real (significant) or could just be chance."""
    err = _check_columns(group_column, value_column)
    if err:
        return err
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return f"'{value_column}' is not numeric, so it cannot be compared."
    if df[group_column].nunique() > 20:
        return f"'{group_column}' has too many unique values to use as groups."
    data = df[[group_column, value_column]].dropna()
    summary = data.groupby(group_column)[value_column].agg(["count", "mean", "median"])
    samples = [g[value_column].values for _, g in data.groupby(group_column)]
    if len(samples) < 2:
        return "Need at least 2 groups to compare."
    if len(samples) == 2:
        _, p = stats.ttest_ind(samples[0], samples[1], equal_var=False)
        test = "Welch t-test"
    else:
        _, p = stats.f_oneway(*samples)
        test = "ANOVA"
    verdict = "SIGNIFICANT (p < 0.05)" if p < 0.05 else "NOT significant (could be chance)"
    return f"{summary.to_string()}\n{test}: p-value = {p:.4f} -> difference is {verdict}"


# ---------- TOOL 4 ----------
def what_drives(target: str) -> str:
    """Finds which other columns are most related to a numeric target column (for example
    Survived or Fare). Shows correlation for numeric columns and the spread of the target
    across categories for text columns."""
    err = _check_columns(target)
    if err:
        return err
    if not pd.api.types.is_numeric_dtype(df[target]):
        return f"'{target}' must be numeric."
    numeric, categorical = [], []
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            r = df[col].corr(df[target])
            if pd.notna(r):
                numeric.append((abs(r), f"{col}: correlation {r:+.2f}"))
        elif df[col].nunique() <= 10:
            m = df.groupby(col)[target].mean()
            categorical.append((m.max() - m.min(),
                f"{col}: average {target} is lowest for '{m.idxmin()}' ({m.min():.2f}) "
                f"and highest for '{m.idxmax()}' ({m.max():.2f})"))
    numeric.sort(reverse=True)
    categorical.sort(reverse=True)
    out = ["Numeric columns (strongest first):"] + [t for _, t in numeric[:5]]
    out += ["Category columns (biggest difference first):"] + [t for _, t in categorical[:5]]
    return "\n".join(out)


# ---------- TOOL 5 ----------
def find_outliers(column: str) -> str:
    """Finds unusually high or low values in a numeric column using the IQR method."""
    err = _check_columns(column)
    if err:
        return err
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"'{column}' is not numeric."
    s = df[column].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = s[(s < low) | (s > high)]
    return (f"Normal range: {low:.2f} to {high:.2f}. "
            f"Outliers found: {len(out)} of {len(s)}. Examples: {out.head(5).tolist()}")


# ---------- TOOL 6 ----------
def plot_chart(chart_type: str, x_column: str, y_column: str = "") -> str:
    """Creates a chart and saves it as chart.png. chart_type must be one of:
    'hist' (distribution of x_column), 'bar' (average of y_column for each x_column category,
    or counts of x_column if y_column is empty), 'scatter' (x_column versus y_column)."""
    err = _check_columns(x_column, y_column)
    if err:
        return err
    plt.figure()
    if chart_type == "hist":
        df[x_column].dropna().plot(kind="hist")
    elif chart_type == "bar":
        if y_column:
            df.groupby(x_column)[y_column].mean().plot(kind="bar")
        else:
            df[x_column].value_counts().plot(kind="bar")
    elif chart_type == "scatter":
        if not y_column:
            return "Scatter needs y_column."
        plt.scatter(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
    else:
        return "chart_type must be 'hist', 'bar' or 'scatter'."
    plt.title(f"{chart_type}: {x_column} {y_column}".strip())
    plt.tight_layout()
    plt.savefig("chart.png")
    plt.close()
    return "Chart saved as chart.png"


# ---------- TOOL 7 ----------
def top_values(column: str, n: int = 10, separator: str = "") -> str:
    """Counts how often each value appears in a column and returns the n most common values
    with their counts and percentages. Use this for questions like 'which country has the most
    titles' or 'how many movies vs TV shows'. If a single cell can hold several values joined
    by a separator (for example 'United States, India' in a country column), pass that
    separator (for example ', ') so each value is counted separately."""
    err = _check_columns(column)
    if err:
        return err
    s = df[column].dropna().astype(str)
    auto_split = False
    if not separator and df[column].nunique() < 0.5 * len(df):
        if s.str.contains(", ").mean() > 0.1:
            separator = ", "
            auto_split = True
    if separator:
        s = s.str.split(separator).explode().str.strip()
        s = s[s != ""]
    counts = s.value_counts().head(int(n))
    total = len(s)
    lines = [f"{value}: {count} ({count / total * 100:.1f}%)" for value, count in counts.items()]
    missing = int(df[column].isna().sum())
    note = " (cells with several comma-separated values were split and each value counted separately; percentages are of mentions)" if separator else ""
    return f"Top {len(lines)} values of '{column}'{note}, rows with missing value: {missing}\n" + "\n".join(lines)

# ---------- TOOL 8 ----------
def describe_column(column: str) -> str:
    """Returns overall summary statistics for one numeric column (count, mean, median, standard
    deviation, min, max, quartiles). Use this for questions about the overall average, median,
    minimum or maximum of a column, without splitting it into groups."""
    err = _check_columns(column)
    if err:
        return err
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"'{column}' is not numeric. Use top_values for text columns."
    s = df[column].dropna()
    return (f"{column}: count={len(s)}, mean={s.mean():.4f}, median={s.median():.4f}, "
            f"std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}, "
            f"25%={s.quantile(0.25):.4f}, 75%={s.quantile(0.75):.4f}")