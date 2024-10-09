import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import seaborn as sns

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

print(f'Train size: {len(train_data)}, Validation size: {len(validation_data)}')

# Chọn các cột số để tính toán trung bình và độ lệch chuẩn
numeric_train_data = train_data.select_dtypes(include=[np.number])

# Tính trung bình và độ lệch chuẩn
means = numeric_train_data.mean()
stds = numeric_train_data.std()

# Hàm chuẩn hóa dữ liệu
def normalize(df, means, stds):
    df['Weight'] = (df['Weight'] - means.Weight) / stds.Weight
    df['Height'] = (df['Height'] - means.Height) / stds.Height
    df['Age'] = (df['Age'] - means.Age) / stds.Age
    if 'BMI' in df.columns:
        df['BMI'] = (df['BMI'] - means.BMI) / stds.BMI

# Chuẩn hóa tập dữ liệu training và validation
normalize(train_data, means, stds)
normalize(validation_data, means, stds)

# Tạo ma trận đặc trưng X và biến mục tiêu y
X_train = train_data[['Age', 'Height', 'Weight']]
y_train = train_data['BMI']
X_validation = validation_data[['Age', 'Height', 'Weight']]
y_validation = validation_data['BMI']

# Khởi tạo và huấn luyện mô hình Ridge
ridge_model = Ridge(alpha=1.0)  # Alpha là tham số điều chỉnh
ridge_model.fit(X_train, y_train)

# Dự đoán trên tập validation
y_pred_train = ridge_model.predict(X_train)
y_pred_validation = ridge_model.predict(X_validation)

# Đánh giá mô hình
def evaluate_model(y_true, y_pred, data_label="Data"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'Evaluation Metrics for {data_label}:')
    print(f'R² Score: {r2}, MSE: {mse}, RMSE: {rmse}')
    return {"R² Score": r2, "MSE": mse, "RMSE": rmse}

# Đánh giá trên tập train
train_metrics = evaluate_model(y_train, y_pred_train, data_label="Training")

# Đánh giá trên tập validation
validation_metrics = evaluate_model(y_validation, y_pred_validation, data_label="Validation")

# Tạo bảng kết quả chỉ cho train và validation
metrics_df = pd.DataFrame({
    "Dataset": ["Training", "Validation"],
    "R² Score": [train_metrics["R² Score"], validation_metrics["R² Score"]],
    "MSE": [train_metrics["MSE"], validation_metrics["MSE"]],
    "RMSE": [train_metrics["RMSE"], validation_metrics["RMSE"]]
})

# Hiển thị bảng kết quả
print(metrics_df)

# Vẽ biểu đồ dự đoán BMI vs thực tế
def plot_predictions(y_true, y_pred, data_label):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue', label=f'Predicted vs Actual {data_label}')
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             color='red', lw=2, label='Perfect Prediction Line')
    plt.xlabel('Actual BMI')
    plt.ylabel('Predicted BMI')
    plt.title(f'Predicted vs Actual BMI ({data_label})')
    plt.legend()
    plt.show()

# Vẽ biểu đồ dự đoán cho tập train
plot_predictions(y_train, y_pred_train, data_label="Training")

# Vẽ biểu đồ dự đoán cho tập validation
plot_predictions(y_validation, y_pred_validation, data_label="Validation")

# Tạo ma trận tương quan cho dữ liệu training
correlation_matrix = train_data[['BMI', 'Height', 'Weight', 'Age']].corr()

# Vẽ biểu đồ heatmap cho ma trận tương quan
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Training Data')
plt.show()

# Hàm cho phép người dùng nhập dữ liệu và dự đoán BMI
def predict_bmi():
    age = float(input("Nhập độ tuổi (Age): "))
    height = float(input("Nhập chiều cao (Height) tính bằng mét: "))
    weight = float(input("Nhập cân nặng (Weight) tính bằng kg: "))
    
    # Chuẩn hóa dữ liệu đầu vào
    normalized_input = np.array([(age - means.Age) / stds.Age,
                                  (height - means.Height) / stds.Height,
                                  (weight - means.Weight) / stds.Weight]).reshape(1, -1)
    
    # Dự đoán BMI
    predicted_bmi = ridge_model.predict(normalized_input)
    print(f"Dự đoán BMI: {predicted_bmi[0]}")

# Gọi hàm dự đoán
predict_bmi()
