import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Đọc dữ liệu từ file CSV
df = pd.read_csv('bmi_data.csv')
df.columns = ['Sex', 'Age', 'Height', 'Weight', 'BMI']

# Xóa các hàng bị thiếu
df = df.dropna()

# Chỉ lấy các cột số
X = df[['Age', 'Height', 'Weight']]
y = df['BMI']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình mạng nơ-ron với Input layer
model = Sequential()
model.add(Input(shape=(3,)))  # Đầu vào có 3 đặc trưng: Age, Height, Weight
model.add(Dense(64, activation='relu'))  # Lớp ẩn đầu tiên có 64 nơ-ron
model.add(Dense(32, activation='relu'))  # Lớp ẩn thứ hai có 32 nơ-ron
model.add(Dense(1))  # Lớp đầu ra dự đoán 1 giá trị (BMI)

# Biên dịch mô hình với hàm mất mát MSE và thuật toán tối ưu Adam
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Đánh giá mô hình
train_loss = model.evaluate(X_train, y_train)
val_loss = model.evaluate(X_val, y_val)

print(f'Training loss: {train_loss}')
print(f'Validation loss: {val_loss}')

# Dự đoán cho tập huấn luyện
y_train_pred = model.predict(X_train)

# Dự đoán cho tập kiểm tra
y_val_pred = model.predict(X_val)

# Tính toán R² và RMSE cho tập kiểm tra
r2 = r2_score(y_val, y_val_pred)
mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)

print(f'R² score: {r2}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# Vẽ đồ thị quá trình huấn luyện
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Vẽ biểu đồ dự đoán cho tập huấn luyện
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')  # Đường chéo
plt.title('Actual vs Predicted BMI for Training Data')
plt.xlabel('Actual BMI')
plt.ylabel('Predicted BMI')
plt.xlim([y_train.min(), y_train.max()])
plt.ylim([y_train.min(), y_train.max()])
plt.grid()
plt.show()

# Hàm dự đoán cho dữ liệu mới
def predict_new_data(age, height, weight):
    # Chuyển đổi dữ liệu mới thành mảng NumPy
    new_data = np.array([[age, height, weight]])
    # Dự đoán BMI
    predicted_bmi = model.predict(new_data)
    return predicted_bmi[0][0]

# Nhập dữ liệu từ người dùng
name = input("Nhập tên: ")
age = float(input("Nhập tuổi: "))
height = float(input("Nhập chiều cao (inches): "))
weight = float(input("Nhập cân nặng (pounds): "))

predicted_bmi = predict_new_data(age, height, weight)
print(f"{name}'s predicted BMI: {predicted_bmi}")
