import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from lab1.gain_ratio import gain_ratio

sea.set()


plt.figure(figsize=(12, 7))
df = pd.read_csv('../data/data_lab_1.csv')
print(df.describe())

numeric_columns = df.select_dtypes(include=[np.number]).columns
numeric_columns_without_brand = [col for col in numeric_columns if not col.startswith('Brand')]
df_numeric = df[numeric_columns_without_brand]
correlation_matrix = df_numeric.corr()


df = pd.get_dummies(df, columns=['Brand'], prefix='Brand')
# sea.heatmap(correlation_matrix.corr(), annot=True, fmt='.2f')
plt.figure(figsize=(10, 8))
scaler = MinMaxScaler()
df[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight', 'Price']] = scaler.fit_transform(
    df[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight', 'Price']])
df['Price_Category'] = pd.cut(df['Price'], bins=[0, 0.2, 0.5, 1], labels=["Low", "Medium", "High"])
print(df.head())
print(df)
df["Price"].hist()


for attribute_column in df.columns:
    if attribute_column != 'Price' and df[attribute_column].dtype in [np.int64, np.float64]:
        res = gain_ratio(df, 'Price_Category', attribute_column)
        print(f"GR: {attribute_column}: {res}")


plt.show()
