import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
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
train_data, validation_data = train_test_split(df, train_size=train_pct, random_state=42)

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

# Hàm khôi phục dữ liệu sau khi chuẩn hóa
def de_normalize(df, means, stds):
    df = df.copy()
    df['Weight'] = df['Weight'] * stds.Weight + means.Weight
    df['Height'] = df['Height'] * stds.Height + means.Height
    df['Age'] = df['Age'] * stds.Age + means.Age
    if 'BMI' in df.columns:
        df['BMI'] = df['BMI'] * stds.BMI + means.BMI
    if 'predictionBMI' in df.columns:
        df['predictionBMI'] = df['predictionBMI'] * stds.BMI + means.BMI
    return df

# Chuẩn hóa tập dữ liệu training và validation
normalize(train_data, means, stds)
normalize(validation_data, means, stds)

# Định nghĩa các mô hình cơ sở
base_models = [
    ('lr', LinearRegression()),
    ('dt', DecisionTreeRegressor()),
    ('rf', RandomForestRegressor(n_estimators=100))
]

# Định nghĩa mô hình meta-learner
meta_model = LinearRegression()

# Tạo mô hình stacking
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Huấn luyện mô hình stacking
X_train = train_data[['Height', 'Weight', 'Age']]
y_train = train_data['BMI']
stacking_model.fit(X_train, y_train)

# Dự đoán trên tập training
train_predictions = stacking_model.predict(X_train)
train_data['predictionBMI'] = train_predictions

# Đánh giá mô hình trên tập train
def evaluate_model(df, means, stds, data_label="Data"):
    df_denorm = de_normalize(df, means, stds)
    if 'BMI' not in df_denorm.columns or df_denorm['BMI'].isnull().all():
        print(f'{data_label} data does not have true BMI values. Skipping evaluation.')
        return None
    r2 = r2_score(df_denorm['BMI'], df_denorm['predictionBMI'])
    mse = mean_squared_error(df_denorm['BMI'], df_denorm['predictionBMI'])
    rmse = np.sqrt(mse)
    return {"R² Score": r2, "MSE": mse, "RMSE": rmse}

# Đánh giá trên tập train
train_metrics = evaluate_model(train_data, means, stds, data_label="Training")

# Dự đoán trên tập validation
X_validation = validation_data[['Height', 'Weight', 'Age']]
y_validation = validation_data['BMI']
validation_predictions = stacking_model.predict(X_validation)
validation_data['predictionBMI'] = validation_predictions

# Đánh giá trên tập validation
validation_metrics = evaluate_model(validation_data, means, stds, data_label="Validation")

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
def plot_predictions(df, means, stds, data_label):
    df_denorm = de_normalize(df, means, stds)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_denorm['BMI'], df_denorm['predictionBMI'], color='blue', label=f'Predicted vs Actual {data_label}')
    plt.plot([df_denorm['BMI'].min(), df_denorm['BMI'].max()], 
             [df_denorm['BMI'].min(), df_denorm['BMI'].max()], 
             color='red', lw=2, label='Perfect Prediction Line')
    plt.xlabel('Actual BMI')
    plt.ylabel('Predicted BMI')
    plt.title(f'Predicted vs Actual BMI ({data_label})')
    plt.legend()
    plt.show()

# Vẽ biểu đồ dự đoán cho tập train
plot_predictions(train_data, means, stds, data_label="Training")

# Vẽ biểu đồ dự đoán cho tập validation
plot_predictions(validation_data, means, stds, data_label="Validation")

# Tạo ma trận tương quan cho dữ liệu training
correlation_matrix = train_data[['BMI', 'Height', 'Weight', 'Age']].corr()

# Vẽ biểu đồ heatmap cho ma trận tương quan
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Training Data')
plt.show()
