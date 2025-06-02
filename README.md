# Task--4
Exploratory Data Analysis (EDA)
# 🔧 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 📂 2. Load Dataset
df = pd.read_csv('your_dataset.csv')  # Replace with your actual file
print("✅ Dataset loaded.")
display(df.head())

# 🧠 3. Basic Info
print("\n📊 Shape:", df.shape)
print("\n🔍 Data Types:\n", df.dtypes)
print("\n❓ Missing Values:\n", df.isnull().sum())
print("\n📈 Summary Statistics:\n")
display(df.describe(include='all'))

# 🧹 4. Data Cleaning
df.drop_duplicates(inplace=True)
# Example: df['column'].fillna(df['column'].mean(), inplace=True)

print("\n🧹 Cleaned data (missing values):\n", df.isnull().sum())

# 🔎 5. Univariate Analysis – Numerical
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_features:
    plt.figure()
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.show()

# 🔎 6. Univariate Analysis – Categorical
categorical_features = df.select_dtypes(include='object').columns
for col in categorical_features:
    plt.figure()
    sns.countplot(data=df, x=col, palette='Set2')
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45)
    plt.show()

# 🔁 7. Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 🧪 8. Bivariate Relationships (Examples)
# Replace with real column names
if 'feature1' in df.columns and 'feature2' in df.columns:
    sns.scatterplot(data=df, x='feature1', y='feature2', hue='target')
    plt.title('Feature1 vs Feature2')
    plt.show()

# 📦 9. Boxplots for Outliers
for col in numerical_features:
    plt.figure()
    sns.boxplot(data=df, x=col, color='lightcoral')
    plt.title(f'Boxplot of {col}')
    plt.show()

# 📊 10. Grouped Statistics (optional)
# Replace 'category_column' with real column
# display(df.groupby('category_column')[numerical_features].mean())
