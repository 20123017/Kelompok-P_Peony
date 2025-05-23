import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

df = pd.read_csv('winequality-red.csv')  
print("✅ Data berhasil dimuat!")

print("\n📌 5 Baris Pertama:")
print(df.head())

print("\n📌 Informasi Data:")
print(df.info())

print("\n📌 Statistik Deskriptif:")
print(df.describe())

print("\n📌 Cek Nilai Kosong:")
print(df.isnull().sum())

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('🔍 Korelasi Antar Fitur')
plt.show()

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📈 Evaluasi Model:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

with open('wine_quality_model.pkl', 'wb') as f:
    pickle.dump(model, f)

sample_data = np.array([[7.4, 0.70, 0.00, 1.9, 0.076,
                         11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])

sample_scaled = scaler.transform(sample_data)
predicted_quality = model.predict(sample_scaled)

print(f"\n🔮 Prediksi kualitas anggur untuk data baru: {predicted_quality[0]:.2f}")
