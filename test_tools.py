import pandas as pd
import tools
tools.set_dataframe(pd.read_csv("train.csv"))
from tools import *

print("--- SCHEMA ---")
print(get_schema())
print("\n--- QUALITY ---")
print(data_quality_audit())
print("\n--- COMPARE ---")
print(compare_groups("Sex", "Fare"))
print("\n--- DRIVERS ---")
print(what_drives("Survived"))
print("\n--- OUTLIERS ---")
print(find_outliers("Fare"))
print("\n--- PLOT ---")
print(plot_chart("bar", "Sex", "Survived"))