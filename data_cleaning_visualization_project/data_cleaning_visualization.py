# Data Cleaning & Visualization Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

# Fill missing values with mean
data.fillna(data.mean(numeric_only=True), inplace=True)

# Remove duplicates
data = data.drop_duplicates()

print("\nData after cleaning:")
print(data.head())

# Histogram
data.hist(figsize=(10,8))
plt.title("Histogram of Dataset Features")
plt.savefig("histogram.png")   # saves image in folder
plt.show()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()

# Boxplot
plt.figure(figsize=(8,6))
sns.boxplot(data=data.select_dtypes(include=['float64','int64']))
plt.title("Boxplot for Outlier Detection")
plt.savefig("boxplot.png")
plt.show()

print("Project Completed Successfully!")