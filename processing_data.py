import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# 1. Menghapus data duplikat
df = df.drop_duplicates()

# 2. Menghapus data yang tidak lengkap (missing values)
df = df.dropna()

# 3. Normalisasi kolom numerik saja
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Cek apakah ada kolom numerik untuk di-normalisasi
if not numerical_cols.empty:
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Menampilkan 5 data pertama setelah preprocessing
print(df.head())

# Simpan hasil preprocessing ke file CSV di direktori saat ini
df.to_csv("mall_custumers.csv", index=False)