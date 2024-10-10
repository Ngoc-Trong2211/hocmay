import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu từ file CSV
df = pd.read_csv('bmi_data.csv')
df.columns = ['Sex', 'Age', 'Height', 'Weight', 'BMI']

# Xóa các hàng bị thiếu
df = df.dropna()

# Hiển thị tổng quan dữ liệu
print(df.head(5))  # Hiển thị 5 hàng đầu tiên
print(df.describe())  # Hiển thị thống kê mô tả của dữ liệu

# Tạo màu sắc dựa trên BMI
colors = [(1 - (BMI - 13) / 14, 0, 0) for BMI in df.BMI.values]
fig, ax = plt.subplots()
ax.scatter(df['Weight'].values, df['Height'].values, c=colors, picker=True)
ax.set_xlabel('Weight')
ax.set_ylabel('Height')
ax.set_title('BMI Distribution')
plt.show()

# Phân chia dữ liệu thành tập train và validation
train_pct = 0.8
train_index = int(len(df) * train_pct)

train_data = df.iloc[:train_index].copy()
validation_data = df.iloc[train_index:].copy()

print(f'train = {len(train_data)},\nvalidation = {len(validation_data)}')

# Chuẩn bị dữ liệu cho mô hình
X_train = train_data[['Height', 'Weight', 'Age']]
y_train = train_data['BMI']
X_val = validation_data[['Height', 'Weight', 'Age']]
y_val = validation_data['BMI']

# Khởi tạo StandardScaler
scaler = StandardScaler()

# Chuẩn hóa dữ liệu
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Khởi tạo mô hình Ridge
ridge_model = Ridge(alpha=1.0)  # alpha là tham số điều chỉnh

# Huấn luyện mô hình
ridge_model.fit(X_train, y_train)

# Dự đoán trên tập validation
validation_predictions = ridge_model.predict(X_val)

# Hàm đánh giá mô hình
def evaluate_model(y_true, y_pred, data_label="Data"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f'{data_label} R² score: {r2}')
    print(f'{data_label} MSE: {mse}')
    print(f'{data_label} RMSE: {rmse}')

# Đánh giá trên tập validation
evaluate_model(y_val, validation_predictions, data_label="Validation")

# Tạo ma trận tương quan
plt.figure(figsize=(10, 8))

# Chọn các cột số để tính toán ma trận tương quan
numeric_df = df.select_dtypes(include=[np.number])

# Tính toán ma trận tương quan
correlation_matrix = numeric_df.corr()  

# Vẽ ma trận tương quan
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Dự đoán trên tập train
y_train_pred = ridge_model.predict(X_train)

# Vẽ biểu đồ dự đoán cho tập huấn luyện
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5, label='Dự đoán')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Đường chéo (y=x)')  # Đường chéo

# Cài đặt các yếu tố trên biểu đồ
plt.title('Actual vs Predicted BMI for Training Data - Ridge')
plt.xlabel('Actual BMI')
plt.ylabel('Predicted BMI')
plt.xlim([y_train.min(), y_train.max()])
plt.ylim([y_train.min(), y_train.max()])
plt.grid(True)
plt.legend()
plt.show()

# Hàm nhập dữ liệu từ người dùng
def get_user_input():
    name = input("Nhập tên: ")
    age = float(input("Nhập tuổi: "))
    height = float(input("Nhập chiều cao (inches): "))
    weight = float(input("Nhập cân nặng (pounds): "))
    return [{'name': name, 'Age': age, 'Height': height, 'Weight': weight}]

# Dự đoán cho dữ liệu mới
def predictBMI_real(data):
    df = pd.DataFrame(data)
    # Chuẩn hóa dữ liệu mới
    new_data = scaler.transform(df[['Height', 'Weight', 'Age']])
    prediction = ridge_model.predict(new_data)
    df['predictionBMI'] = prediction
    return df

# Nhập dữ liệu từ người dùng và dự đoán
new_data = get_user_input()
predicted_df = predictBMI_real(new_data)

# Hiển thị kết quả dự đoán
print(predicted_df[['name', 'Age', 'Height', 'Weight', 'predictionBMI']])
