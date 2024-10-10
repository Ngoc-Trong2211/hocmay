import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.metrics import r2_score, mean_squared_error

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

# Chọn các cột số để tính toán trung bình và độ lệch chuẩn
numeric_train_data = train_data.select_dtypes(include=[np.number])

# Khởi tạo trọng số ngẫu nhiên
def reset():
    global w1, w2, w3, bias
    w1 = np.random.randn()
    w2 = np.random.randn()
    w3 = np.random.randn()
    bias = np.random.randn()

reset()

# Chuẩn hóa dữ liệu
def normalize(df, means, stds):
    df['Weight'] = (df['Weight'] - means.Weight) / stds.Weight
    df['Height'] = (df['Height'] - means.Height) / stds.Height
    df['Age'] = (df['Age'] - means.Age) / stds.Age
    if 'BMI' in df.columns:
        df['BMI'] = (df['BMI'] - means.BMI) / stds.BMI

# Khôi phục dữ liệu sau khi chuẩn hóa
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

# Tính trung bình và độ lệch chuẩn
means = numeric_train_data.mean()
stds = numeric_train_data.std()

# Chuẩn hóa tập dữ liệu training và validation
normalize(train_data, means, stds)
normalize(validation_data, means, stds)

# Hàm dự đoán BMI
def predict_BMI(df):
    pred = w1 * df['Height'] + w2 * df['Weight'] + w3 * df['Age'] + bias
    df['predictionBMI'] = pred
    return df

# Hàm tính toán loss
def calculate_loss(df):
    return np.square(df['predictionBMI'] - df['BMI']).mean()

# Hàm tính gradient
def calculate_gradients(df):
    diff = df['predictionBMI'] - df['BMI']
    dw1 = 2 * diff * df['Height']
    dw2 = 2 * diff * df['Weight']
    dw3 = 2 * diff * df['Age']
    dbias = 2 * diff
    dw1, dw2, dw3, dbias = dw1.mean(), dw2.mean(), dw3.mean(), dbias.mean()
    return dw1, dw2, dw3, dbias

# Hàm train mô hình
def train(learning_rate=0.01):
    global w1, w2, w3, bias
    preddf = predict_BMI(train_data)
    dw1, dw2, dw3, dbias = calculate_gradients(preddf)
    w1 -= dw1 * learning_rate
    w2 -= dw2 * learning_rate
    w3 -= dw3 * learning_rate
    bias -= dbias * learning_rate
    return calculate_loss(preddf)

# Training mô hình
learning_rate = 0.01
for i in tqdm(range(300)):
    loss = train(learning_rate)
    time.sleep(0.01)
    if i % 20 == 0:
        print(f'epoch: {i}, loss = {loss}')

# Hàm đánh giá mô hình
def evaluate_model(df, means, stds, data_label="Data"):
    df_denorm = de_normalize(df, means, stds)
    r2 = r2_score(df_denorm['BMI'], df_denorm['predictionBMI'])
    mse = mean_squared_error(df_denorm['BMI'], df_denorm['predictionBMI'])
    rmse = np.sqrt(mse)
    
    print(f'{data_label} R² score: {r2}')
    print(f'{data_label} MSE: {mse}')
    print(f'{data_label} RMSE: {rmse}')

# Đánh giá trên tập train
evaluate_model(train_data, means, stds, data_label="Training")

# Đánh giá trên tập validation
validation_predictions = predict_BMI(validation_data)
evaluate_model(validation_predictions, means, stds, data_label="Validation")

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
    normalize(df, means, stds)
    df = predict_BMI(df)
    return de_normalize(df, means, stds)

# Nhập dữ liệu từ người dùng và dự đoán
new_data = get_user_input()
predicted_df = predictBMI_real(new_data)

# Hiển thị kết quả dự đoán
print(predicted_df[['name', 'Age', 'Height', 'Weight', 'predictionBMI']])