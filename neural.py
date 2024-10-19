from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Tải dữ liệu BMI và xử lý
du_lieu_bmi = pd.read_csv('bmi_data.csv').dropna()
dac_trung = du_lieu_bmi[['Age', 'Height(Inches)', 'Weight(Pounds)']]
muc_tieu = du_lieu_bmi['BMI']

# 1. Đánh giá dữ liệu đầu vào
print("Giá trị thiếu trong dữ liệu:")
print(du_lieu_bmi.isnull().sum())

# Xử lý ngoại lai bằng phương pháp IQR
Q1 = du_lieu_bmi[['Age', 'Height(Inches)', 'Weight(Pounds)']].quantile(0.25)
Q3 = du_lieu_bmi[['Age', 'Height(Inches)', 'Weight(Pounds)']].quantile(0.75)
IQR = Q3 - Q1

# Xóa các ngoại lai
du_lieu_bmi = du_lieu_bmi[~((du_lieu_bmi[['Age', 'Height(Inches)', 'Weight(Pounds)']] < (Q1 - 1.5 * IQR)) | 
                             (du_lieu_bmi[['Age', 'Height(Inches)', 'Weight(Pounds)']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# In lại số lượng mẫu sau khi loại bỏ ngoại lai
print("Số mẫu sau khi loại bỏ ngoại lai:", du_lieu_bmi.shape[0])

# Cập nhật lại biến sau khi loại bỏ ngoại lai
dac_trung = du_lieu_bmi[['Age', 'Height(Inches)', 'Weight(Pounds)']]
muc_tieu = du_lieu_bmi['BMI']

# Phân phối đặc trưng
plt.figure(figsize=(15, 5))

# Tuổi
plt.subplot(1, 3, 1)
plt.hist(du_lieu_bmi['Age'], bins=30, color='skyblue', edgecolor='black')
plt.title('Phân phối tuổi')
plt.xlabel('Tuổi')
plt.ylabel('Tần suất')

# Chiều cao
plt.subplot(1, 3, 2)
plt.hist(du_lieu_bmi['Height(Inches)'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Phân phối chiều cao')
plt.xlabel('Chiều cao (Inches)')
plt.ylabel('Tần suất')

# Cân nặng
plt.subplot(1, 3, 3)
plt.hist(du_lieu_bmi['Weight(Pounds)'], bins=30, color='salmon', edgecolor='black')
plt.title('Phân phối cân nặng')
plt.xlabel('Cân nặng (Pounds)')
plt.ylabel('Tần suất')

plt.tight_layout()
plt.show()

# Boxplot kiểm tra outliers
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.boxplot(du_lieu_bmi['Age'])
plt.title('Boxplot cho tuổi')

plt.subplot(1, 3, 2)
plt.boxplot(du_lieu_bmi['Height(Inches)'])
plt.title('Boxplot cho chiều cao')

plt.subplot(1, 3, 3)
plt.boxplot(du_lieu_bmi['Weight(Pounds)'])
plt.title('Boxplot cho cân nặng')

plt.tight_layout()
plt.show()

# 2. Chia dữ liệu
X_train, X_temp, y_train, y_temp = train_test_split(dac_trung, muc_tieu, test_size=0.3, random_state=43)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

# 3. Chuẩn hóa dữ liệu
scaler = MinMaxScaler().fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_val = scaler.transform(X_val)
scaled_X_test = scaler.transform(X_test)

# 4. Các tham số cho GridSearchCV
tham_so_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'max_iter': [1000, 3000, 5000]
}

# 5. Tìm tham số tối ưu
grid_search = GridSearchCV(MLPRegressor(random_state=1), tham_so_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(scaled_X_train, y_train)

# In ra tham số tốt nhất
print("Tham số tối ưu tìm được:")
print(grid_search.best_params_)

# 6. Dự đoán và tính toán chỉ số
mlp_tot_nhat = grid_search.best_estimator_

# Dự đoán cho từng tập dữ liệu
y_train_pred = mlp_tot_nhat.predict(scaled_X_train)
y_val_pred = mlp_tot_nhat.predict(scaled_X_val)
y_test_pred = mlp_tot_nhat.predict(scaled_X_test)

# Tính toán các chỉ số cho từng tập dữ liệu
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# In ra kết quả
print("Kết quả cho tập huấn luyện:")
print(f"MSE: {mse_train}, MAE: {mae_train}, R²: {r2_train}")

print("Kết quả cho tập xác thực:")
print(f"MSE: {mse_val}, MAE: {mae_val}, R²: {r2_val}")

print("Kết quả cho tập kiểm tra:")
print(f"MSE: {mse_test}, MAE: {mae_test}, R²: {r2_test}")

# 7. Vẽ biểu đồ so sánh giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(15, 5))

# Tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.title('Tập huấn luyện: Thực tế vs Dự đoán')
plt.xlabel('BMI Thực tế')
plt.ylabel('BMI Dự đoán')
plt.grid()

# Tập xác thực
plt.subplot(1, 3, 2)
plt.scatter(y_val, y_val_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.title('Tập xác thực: Thực tế vs Dự đoán')
plt.xlabel('BMI Thực tế')
plt.ylabel('BMI Dự đoán')
plt.grid()

# Tập kiểm tra
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Tập kiểm tra: Thực tế vs Dự đoán')
plt.xlabel('BMI Thực tế')
plt.ylabel('BMI Dự đoán')
plt.grid()

plt.tight_layout()
plt.show()
