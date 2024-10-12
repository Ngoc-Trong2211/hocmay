import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_squared_error

# Tải dữ liệu cổ phiếu
stock_data = pd.read_csv('bmi_data.csv')

# Loại bỏ cột 'Sex'
stock_data.drop(columns=['Sex'], inplace=True)

# Kiểm tra và thay thế hoặc loại bỏ NaN
# Thay thế các giá trị NaN bằng giá trị trung bình của các cột
stock_data.fillna(stock_data.mean(), inplace=True)

# Đặc trưng và target
features = stock_data[['Age', 'Height(Inches)', 'Weight(Pounds)']]
target = stock_data['BMI']

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=43)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo RidgeCV với danh sách các giá trị alpha
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
ridge_cv.fit(X_train_scaled, y_train)

# Loss function: Sử dụng Mean Squared Error (MSE) làm hàm loss
mse_train = []
mse_test = []

# Dự đoán và tính toán MSE cho mỗi alpha
for alpha in alphas:
    ridge = RidgeCV(alphas=[alpha])
    ridge.fit(X_train_scaled, y_train)

    # Dự đoán cho tập train và test
    y_train_pred = ridge.predict(X_train_scaled)
    y_test_pred = ridge.predict(X_test_scaled)

    # Tính toán MSE
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))

# Vẽ biểu đồ loss (MSE) theo các giá trị alpha
plt.figure(figsize=(10, 6))
plt.plot(alphas, mse_train, label='MSE on Training set', marker='o')
plt.plot(alphas, mse_test, label='MSE on Test set', marker='o')

plt.xscale('log')  # Sử dụng scale log cho các giá trị alpha
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Loss Function (MSE) for Ridge Regression')
plt.legend()
plt.grid(True)
plt.show()
