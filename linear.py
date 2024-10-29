from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Tải dữ liệu cổ phiếu
dulieu = pd.read_csv('bmi_data.csv')

# Xem thông tin tập dữ liệu
print(dulieu.info())

# Kiểm tra giá trị null trong tập dữ liệu
null_values = dulieu.isnull().sum()
print("Số lượng giá trị null trong từng cột:")
print(null_values)

dulieu = dulieu.select_dtypes(include=[np.number])  # Chỉ giữ lại các biến số

# Xóa các hàng bị thiếu
dulieu = dulieu.dropna()

features = dulieu[['Age', 'Height(Inches)', 'Weight(Pounds)']]
target = dulieu['BMI']

# Chia dữ liệu thành 70% train và 30% còn lại
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)

# Chia tập còn lại thành 15% test và 15% validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

# Huấn luyện mô hình
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Dự đoán
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)
y_val_pred = linreg.predict(X_val)

# Hàm tính toán loss function (MSE và RMSE)
def calculate_loss(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse

# Tính toán MSE và RMSE cho cả tập train, test và validation
mse_train, rmse_train = calculate_loss(y_train, y_train_pred)
mse_test, rmse_test = calculate_loss(y_test, y_test_pred)
mse_val, rmse_val = calculate_loss(y_val, y_val_pred)

# Tính toán R² cho cả tập train, test và validation
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_val = r2_score(y_val, y_val_pred)

# Kết quả MSE, RMSE và R²
results = pd.DataFrame({
    'Dataset': ['Train', 'Test', 'Validation'],
    'R²': [r2_train, r2_test, r2_val],
    'MSE': [mse_train, mse_test, mse_val],
    'RMSE': [rmse_train, rmse_test, rmse_val]
})

# Hiển thị bảng
print(results)

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

import seaborn as sns

# Tính toán ma trận tương quan
correlation_matrix = dulieu.corr()

# Vẽ biểu đồ ma trận tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar=True)
plt.title('Correlation Matrix')
plt.show()

# In ra mối tương quan với BMI
bmi_correlation = correlation_matrix['BMI']
print(bmi_correlation)

import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập kích thước cho các biểu đồ
plt.figure(figsize=(15, 5))

# Biểu đồ phân phối cho tuổi
plt.subplot(1, 3, 1)
sns.histplot(dulieu['Age'], bins=30, kde=True, color='blue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Biểu đồ phân phối cho cân nặng
plt.subplot(1, 3, 2)
sns.histplot(dulieu['Weight(Pounds)'], bins=30, kde=True, color='green')
plt.title('Distribution of Weight')
plt.xlabel('Weight (Pounds)')
plt.ylabel('Frequency')

# Biểu đồ phân phối cho chiều cao
plt.subplot(1, 3, 3)
sns.histplot(dulieu['Height(Inches)'], bins=30, kde=True, color='orange')
plt.title('Distribution of Height')
plt.xlabel('Height (Inches)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Lọc các cột số
dulieu_numeric = dulieu.select_dtypes(include=[np.number])

# Tính toán giá trị trung bình, trung vị và mode cho từng cột
mean_values = dulieu_numeric.mean()  # Giá trị trung bình
median_values = dulieu_numeric.median()  # Trung vị
mode_values = dulieu_numeric.mode().iloc[0]  # Mốt (mode trả về nhiều giá trị, lấy hàng đầu tiên)

# Hiển thị kết quả
print("Giá trị trung bình (Mean):")
print(mean_values)

print("\nTrung vị (Median):")
print(median_values)

print("\nMốt (Mode):")
print(mode_values)

import scipy.stats as stats
import matplotlib.pyplot as plt

# QQ plot để kiểm tra phân phối chuẩn
plt.figure(figsize=(6,6))
stats.probplot(dulieu_numeric['BMI'], dist="norm", plot=plt)
plt.title("QQ Plot for BMI")
plt.show()

# Kiểm định Shapiro-Wilk
shapiro_test = stats.shapiro(dulieu_numeric['BMI'])
print(f"Shapiro-Wilk Test: Statistic={shapiro_test[0]}, p-value={shapiro_test[1]}")

if shapiro_test[1] > 0.05:
    print("Dữ liệu có phân phối chuẩn (theo kiểm định Shapiro-Wilk).")
else:
    print("Dữ liệu không có phân phối chuẩn (theo kiểm định Shapiro-Wilk).")
