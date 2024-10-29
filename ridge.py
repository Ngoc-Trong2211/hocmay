import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Tải dữ liệu
dulieu = pd.read_csv('bmi_data.csv')

# Loại bỏ cột 'Sex' nếu có
if 'Sex' in dulieu.columns:
    dulieu.drop(columns=['Sex'], inplace=True)

# Kiểm tra và thay thế hoặc loại bỏ NaN
dulieu.fillna(dulieu.mean(), inplace=True)

# Kiểm tra phân phối của các cột BMI, Height, Weight
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.histplot(dulieu['BMI'], kde=True)
plt.title('Phân phối của BMI')

plt.subplot(1, 3, 2)
sns.histplot(dulieu['Height(Inches)'], kde=True)
plt.title('Phân phối của Height')

plt.subplot(1, 3, 3)
sns.histplot(dulieu['Weight(Pounds)'], kde=True)
plt.title('Phân phối của Weight')

plt.show()

# Hàm xử lý ngoại lai dựa trên IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Loại bỏ các giá trị ngoại lai trong cột 'BMI', 'Height(Inches)' và 'Weight(Pounds)'
dulieu = remove_outliers_iqr(dulieu, 'BMI')
dulieu = remove_outliers_iqr(dulieu, 'Height(Inches)')
dulieu = remove_outliers_iqr(dulieu, 'Weight(Pounds)')

# Đặc trưng và target
features = dulieu[['Age', 'Height(Inches)', 'Weight(Pounds)']]
target = dulieu['BMI']

# Chia dữ liệu thành 70% train và 30% còn lại
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)

# Chia tập còn lại thành 15% test và 15% validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Tạo dải alpha từ 10^-4 đến 10^4
alphas = np.logspace(-4, 4, 100)

# Khởi tạo RidgeCV với danh sách các giá trị alpha
ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
ridge_cv.fit(X_train_scaled, y_train)

# In ra giá trị alpha tối ưu
best_alpha = ridge_cv.alpha_
print(f'Giá trị alpha tối ưu: {best_alpha}')

# Dự đoán cho tập train, validation và test
y_train_pred = ridge_cv.predict(X_train_scaled)
y_val_pred = ridge_cv.predict(X_val_scaled)
y_test_pred = ridge_cv.predict(X_test_scaled)

# Tính toán các chỉ số hiệu suất
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse

# Tính toán cho các tập train, validation và test
train_metrics = calculate_metrics(y_train, y_train_pred)
val_metrics = calculate_metrics(y_val, y_val_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

# Tạo bảng kết quả
metrics_df = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'R²': [ridge_cv.score(X_train_scaled, y_train), ridge_cv.score(X_val_scaled, y_val), ridge_cv.score(X_test_scaled, y_test)],
    'MSE': [train_metrics[0], val_metrics[0], test_metrics[0]],
    'RMSE': [train_metrics[1], val_metrics[1], test_metrics[1]]
})

print(metrics_df)

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

# Vẽ biểu đồ độ chênh lệch (Residuals)
residuals = y_test - y_test_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted BMI')
plt.xlabel('Predicted BMI')
plt.ylabel('Residuals')
plt.grid()
plt.show()

# # Tải dữ liệu
# dulieu = pd.read_csv('bmi_data.csv')

# # Kiểm tra phân phối BMI
# plt.figure(figsize=(10, 6))
# sns.histplot(dulieu['BMI'], bins=30, kde=True, color='blue')
# plt.title('Phân phối của BMI')
# plt.xlabel('BMI')
# plt.ylabel('Tần suất')
# plt.grid()
# plt.show()

# # Loại bỏ cột 'Sex' (nếu có)
# dulieu.drop(columns=['Sex'], inplace=True, errors='ignore')

# # Kiểm tra và thay thế hoặc loại bỏ NaN
# dulieu.fillna(dulieu.mean(), inplace=True)

# # Đặc trưng và target
# features = dulieu[['Age', 'Height(Inches)', 'Weight(Pounds)']]
# target = dulieu['BMI']

# # Phân tích các điểm dữ liệu có BMI trong khoảng 19-20
# bmi_range = dulieu[(dulieu['BMI'] >= 19) & (dulieu['BMI'] <= 20)]
# print("Số lượng điểm dữ liệu trong khoảng BMI từ 19 đến 20:", bmi_range.shape[0])

# # Kiểm tra phân phối chiều cao và cân nặng cho các điểm trong khoảng này
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# sns.histplot(bmi_range['Height(Inches)'], bins=15, kde=True, color='orange')
# plt.title('Phân phối chiều cao (trong khoảng BMI 19-20)')
# plt.xlabel('Chiều cao (Inches)')
# plt.ylabel('Tần suất')
# plt.grid()

# plt.subplot(1, 2, 2)
# sns.histplot(bmi_range['Weight(Pounds)'], bins=15, kde=True, color='green')
# plt.title('Phân phối cân nặng (trong khoảng BMI 19-20)')
# plt.xlabel('Cân nặng (Pounds)')
# plt.ylabel('Tần suất')
# plt.grid()

# plt.tight_layout()
# plt.show()

# # Chia dữ liệu thành 70% train và 30% còn lại
# X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)

# # Chia tập còn lại thành 15% test và 15% validation
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

# # Chuẩn hóa dữ liệu
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)

# # Khởi tạo RidgeCV
# alphas = np.logspace(-6, 2, 100)
# ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
# ridge_cv.fit(X_train_scaled, y_train)

# # Dự đoán cho tập train, validation và test
# y_train_pred = ridge_cv.predict(X_train_scaled)
# y_val_pred = ridge_cv.predict(X_val_scaled)
# y_test_pred = ridge_cv.predict(X_test_scaled)

# # Vẽ biểu đồ so sánh giữa giá trị thực tế và giá trị dự đoán
# plt.figure(figsize=(18, 6))

# # Biểu đồ cho tập train
# plt.subplot(1, 3, 1)
# plt.scatter(y_train, y_train_pred, color='blue', alpha=0.6)
# plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--')
# plt.title('Actual vs Predicted for Training Set')
# plt.xlabel('Actual BMI')
# plt.ylabel('Predicted BMI')
# plt.grid()

# # Biểu đồ cho tập validation
# plt.subplot(1, 3, 2)
# plt.scatter(y_val, y_val_pred, color='orange', alpha=0.6)
# plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
# plt.title('Actual vs Predicted for Validation Set')
# plt.xlabel('Actual BMI')
# plt.ylabel('Predicted BMI')
# plt.grid()

# # Biểu đồ cho tập test
# plt.subplot(1, 3, 3)
# plt.scatter(y_test, y_test_pred, color='green', alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
# plt.title('Actual vs Predicted for Test Set')
# plt.xlabel('Actual BMI')
# plt.ylabel('Predicted BMI')
# plt.grid()

# plt.tight_layout()
# plt.show()

# # Vẽ biểu đồ residuals
# residuals = y_test - y_test_pred
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test_pred, residuals, alpha=0.6)
# plt.axhline(0, color='red', linestyle='--')
# plt.title('Residuals vs Predicted BMI')
# plt.xlabel('Predicted BMI')
# plt.ylabel('Residuals')
# plt.grid()
# plt.show()

