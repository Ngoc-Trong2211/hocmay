from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Tải dữ liệu cổ phiếu
dulieu = pd.read_csv('bmi_data.csv')

# Xóa các hàng bị thiếu
stock_data = dulieu.dropna()

features = dulieu[['Age', 'Height(Inches)', 'Weight(Pounds)']]
target = dulieu['BMI']

# Chia dữ liệu thành 70% train và 30% còn lại
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)

# Chia tập còn lại thành 15% test và 15% validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

# Huấn luyện mô hình
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print('Linear model intercept b is: {}'.format(linreg.intercept_))
print('Linear model coeff a is: {}'.format(linreg.coef_))

# Dự đoán
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)
y_val_pred = linreg.predict(X_val)

# Tính toán RMSE cho cả tập train, test và validation
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))


# Kết quả MSE và RMSE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mse_val = mean_squared_error(y_val, y_val_pred)

# Tính toán R² cho cả tập train, test và validation
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_val = r2_score(y_val, y_val_pred)

# Tạo DataFrame để hiển thị các giá trị
results = pd.DataFrame({
    'Dataset': ['Train', 'Test', 'Validation'],
    'R²': [r2_train, r2_test, r2_val],
    'MSE': [mse_train, mse_test, mse_val],
    'RMSE': [rmse_train, rmse_test, rmse_val]
})

# Hiển thị bảng
print(results)

# Dự đoán trên tập validation
y_val_pred = linreg.predict(X_val)

# Vẽ biểu đồ so sánh giữa giá trị thực tế và giá trị dự đoán trên tập train, test và validation
plt.figure(figsize=(14, 6))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, alpha=0.7, color='blue')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')  # Đường chuẩn
plt.xlabel('Actual BMI (Train)')
plt.ylabel('Predicted BMI (Train)')
plt.title('Actual vs Predicted BMI (Train)')

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_test_pred, alpha=0.7, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Đường chuẩn
plt.xlabel('Actual BMI (Test)')
plt.ylabel('Predicted BMI (Test)')
plt.title('Actual vs Predicted BMI (Test)')

# Biểu đồ cho tập validation
plt.subplot(1, 3, 3)
plt.scatter(y_val, y_val_pred, alpha=0.7, color='orange')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')  # Đường chuẩn
plt.xlabel('Actual BMI (Validation)')
plt.ylabel('Predicted BMI (Validation)')
plt.title('Actual vs Predicted BMI (Validation)')

plt.tight_layout()
plt.show()