from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Tải dữ liệu BMI
du_lieu_bmi = pd.read_csv('bmi_data.csv')

# Xóa các hàng bị thiếu
du_lieu_bmi = du_lieu_bmi.dropna()

# Chọn đặc trưng và mục tiêu
dac_trung = du_lieu_bmi[['Age', 'Height(Inches)', 'Weight(Pounds)']]
muc_tieu = du_lieu_bmi['BMI']

# Chia dữ liệu thành 70% train và 30% còn lại
X_train, X_temp, y_train, y_temp = train_test_split(dac_trung, muc_tieu, test_size=0.3, random_state=43)

# Chia tập còn lại thành 15% kiểm tra và 15% xác thực
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
scaled_X_val = scaler.transform(X_val)

# Các tham số cho GridSearchCV
tham_so_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'max_iter': [1000, 3000, 5000]
}

# Khởi tạo MLP Regressor
mlp = MLPRegressor(random_state=1)

# GridSearchCV để tìm kiếm tham số tối ưu
grid_search = GridSearchCV(mlp, tham_so_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(scaled_X_train, y_train)

# In ra tham số tốt nhất
print("Các tham số tốt nhất tìm được:", grid_search.best_params_)

# Huấn luyện lại mô hình với tham số tốt nhất
mlp_tot_nhat = grid_search.best_estimator_

# Dự đoán
y_train_pred = mlp_tot_nhat.predict(scaler.transform(X_train))
y_val_pred = mlp_tot_nhat.predict(scaler.transform(X_val))
y_test_pred = mlp_tot_nhat.predict(scaler.transform(X_test))

# Tính toán các chỉ số
r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

rmse_train = np.sqrt(mse_train)
rmse_val = np.sqrt(mse_val)
rmse_test = np.sqrt(mse_test)

# Tạo DataFrame để in ra dưới dạng bảng
ket_qua = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'R²': [r2_train, r2_val, r2_test],
    'MSE': [mse_train, mse_val, mse_test],
    'RMSE': [rmse_train, rmse_val, rmse_test]
})

# In bảng kết quả
print(ket_qua)

# Vẽ biểu đồ cho tập huấn luyện
plt.figure(figsize=(15, 5))

# Tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.title('Tập huấn luyện: Thực tế vs Dự đoán')
plt.xlabel('BMI Thực tế')
plt.ylabel('BMI Dự đoán')
plt.grid()

# Tập xác thực
plt.subplot(1, 3, 2)
plt.scatter(y_val, y_val_pred, color='green', alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.title('Tập xác thực: Thực tế vs Dự đoán')
plt.xlabel('BMI Thực tế')
plt.ylabel('BMI Dự đoán')
plt.grid()

# Tập kiểm tra
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_test_pred, color='orange', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Tập kiểm tra: Thực tế vs Dự đoán')
plt.xlabel('BMI Thực tế')
plt.ylabel('BMI Dự đoán')
plt.grid()

plt.tight_layout()
plt.show()
