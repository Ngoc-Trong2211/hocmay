from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Tải dữ liệu BMI
stock_data = pd.read_csv('bmi_data.csv')

# Xóa các hàng bị thiếu
stock_data = stock_data.dropna()

# Xác định các biến đặc trưng và nhãn
features = stock_data[['Age', 'Height(Inches)', 'Weight(Pounds)']]
target = stock_data['BMI']

# Chia dữ liệu thành 70% cho tập huấn luyện và 30% còn lại
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_val = scaler.transform(X_val)
scaled_X_test = scaler.transform(X_test)

# Khởi tạo RidgeCV để tìm alpha tốt nhất
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], store_cv_results=True)
ridge.fit(scaled_X_train, y_train)
best_alpha = ridge.alpha_
print("Best alpha:", best_alpha)

# Khởi tạo các mô hình cơ bản
mlp = MLPRegressor(random_state=1, max_iter=3000)
linreg = LinearRegression()

# Khởi tạo mô hình Stacking với mô hình cuối cùng là Ridge Regression
stacking_model = StackingRegressor(
    estimators=[('mlp', mlp), ('ridge', ridge), ('linreg', linreg)],
    final_estimator=RidgeCV(alphas=[best_alpha], store_cv_results=True)
)

# Huấn luyện mô hình Stacking
stacking_model.fit(scaled_X_train, y_train)

# Dự đoán trên tập train, validation và test
y_train_pred = stacking_model.predict(scaled_X_train)
y_val_pred = stacking_model.predict(scaled_X_val)
y_test_pred = stacking_model.predict(scaled_X_test)

# Tính toán R², MSE và RMSE cho từng tập dữ liệu
results = {
    'Dataset': ['Train', 'Validation', 'Test'],
    'R2 Score': [
        r2_score(y_train, y_train_pred),
        r2_score(y_val, y_val_pred),
        r2_score(y_test, y_test_pred)
    ],
    'MSE': [
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_val, y_val_pred),
        mean_squared_error(y_test, y_test_pred)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_train, y_train_pred)),
        np.sqrt(mean_squared_error(y_val, y_val_pred)),
        np.sqrt(mean_squared_error(y_test, y_test_pred))
    ]
}

# Tạo DataFrame từ kết quả
results_df = pd.DataFrame(results)

# Hiển thị bảng kết quả
print(results_df)

# Tạo một biểu đồ với 3 subplot
plt.figure(figsize=(18, 6))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, alpha=0.7, color='blue')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')  # Đường chuẩn
plt.xlabel('Actual BMI (Train)')
plt.ylabel('Predicted BMI (Train)')
plt.title('Actual vs Predicted BMI (Train)')
plt.xlim([min(y_train), max(y_train)])
plt.ylim([min(y_train), max(y_train)])
plt.grid()

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_test_pred, alpha=0.7, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Đường chuẩn
plt.xlabel('Actual BMI (Test)')
plt.ylabel('Predicted BMI (Test)')
plt.title('Actual vs Predicted BMI (Test)')
plt.xlim([min(y_test), max(y_test)])
plt.ylim([min(y_test), max(y_test)])
plt.grid()

# Biểu đồ cho tập validation
plt.subplot(1, 3, 3)
plt.scatter(y_val, y_val_pred, alpha=0.7, color='blue')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')  # Đường chuẩn
plt.xlabel('Actual BMI (Validation)')
plt.ylabel('Predicted BMI (Validation)')
plt.title('Actual vs Predicted BMI (Validation)')
plt.xlim([min(y_val), max(y_val)])
plt.ylim([min(y_val), max(y_val)])
plt.grid()

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
