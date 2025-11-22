import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
df_orig = pd.read_csv('data/akc.csv')
#df_kept = df_orig[['height', 'weight', 'size', 'longevity']]
#df_kept = df_orig[['height', 'weight', 'size']]
df_kept = df_orig[['height', 'weight']]
df = df_kept.dropna()

# --- Basic info ---
print("Shape:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nSummary statistics:\n", df.describe(include='all'))
print("\nMissing values:\n", df.isnull().sum())

# --- Histograms ---
print("\nPlotting histograms...")
df.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Histograms of numerical columns", fontsize=14)
plt.tight_layout()
plt.show()

# --- Pairplot for pairwise relationships ---
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_cols) > 1:
    print("\nPlotting pairwise relationships (pairplot)...")
    sns.pairplot(df[numeric_cols], diag_kind='hist')
    plt.suptitle("Pairwise relationships", y=1.02)
    plt.show()

# --- Correlation heatmap ---
print("\nPlotting correlation heatmap...")
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- Scatter plots between selected pairs (example) ---
if len(numeric_cols) >= 2:
    print("\nPlotting scatter plots for first few variable pairs...")
    for i in range(min(3, len(numeric_cols) - 1)):
        plt.figure()
        sns.scatterplot(x=df[numeric_cols[i]], y=df[numeric_cols[i+1]])
        plt.title(f"Scatter: {numeric_cols[i]} vs {numeric_cols[i+1]}")
        plt.show()
        
        
height_col = 'height'
weight_col = 'weight'
class_col = 'size'

print('**********')
print(df.columns)
print('**********')

if not all(col in df.columns for col in [height_col, weight_col, class_col]):
    raise ValueError(f"Expected columns '{height_col}', '{weight_col}', and '{class_col}' not found in CSV.")

# --- Plot height vs weight ---
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x=height_col,
    y=weight_col,
    hue=class_col,
    palette='viridis',
    s=80,
    edgecolor='black'
)

plt.title("Dog Height vs. Weight by Size Class")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.legend(title="Size Class")
plt.grid(True)
plt.tight_layout()
plt.show()