import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_squared_error

# Tải dữ liệu
stock_data = pd.read_csv('bmi_data.csv')

# Loại bỏ cột 'Sex'
stock_data.drop(columns=['Sex'], inplace=True)

# Kiểm tra và thay thế hoặc loại bỏ NaN
stock_data.fillna(stock_data.mean(), inplace=True)

# Đặc trưng và target
features = stock_data[['Age', 'Height(Inches)', 'Weight(Pounds)']]
target = stock_data['BMI']

# Chia dữ liệu thành train, validation và test
X_train_val, X_test, y_train_val, y_test = train_test_split(features, target, test_size=0.3, random_state=43)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=43)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo RidgeCV với danh sách các giá trị alpha
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
ridge_cv.fit(X_train_scaled, y_train)

# Dự đoán cho tập train, validation và test
y_train_pred = ridge_cv.predict(X_train_scaled)
y_val_pred = ridge_cv.predict(X_val_scaled)
y_test_pred = ridge_cv.predict(X_test_scaled)

# Vẽ biểu đồ so sánh giữa giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(18, 6))

# Biểu đồ cho tập train
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted for Training Set')
plt.xlabel('Actual BMI')
plt.ylabel('Predicted BMI')
plt.grid()

# Biểu đồ cho tập validation
plt.subplot(1, 3, 2)
plt.scatter(y_val, y_val_pred, color='orange', alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted for Validation Set')
plt.xlabel('Actual BMI')
plt.ylabel('Predicted BMI')
plt.grid()

# Biểu đồ cho tập test
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_test_pred, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted for Test Set')
plt.xlabel('Actual BMI')
plt.ylabel('Predicted BMI')
plt.grid()

plt.tight_layout()
plt.show()
