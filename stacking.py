from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Tải dữ liệu BMI
stock_data = pd.read_csv('bmi_data.csv')

# Xóa các hàng bị thiếu
stock_data = stock_data.dropna()

features = stock_data[['Age', 'Height(Inches)', 'Weight(Pounds)']]
target = stock_data['BMI']

print(type(stock_data))  # Xác nhận kiểu dữ liệu
print(stock_data.head())

# Chia dữ liệu thành 70% cho tập huấn luyện và 30% còn lại
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
scaled_X_val = scaler.transform(X_val)

# Khởi tạo các mô hình cơ bản
mlp = MLPRegressor(random_state=1, max_iter=3000)
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], store_cv_results=True)  # Sửa ở đây
linreg = LinearRegression()

# Khởi tạo mô hình Stacking với mô hình cuối cùng là Ridge Regression
stacking_model = StackingRegressor(
    estimators=[('mlp', mlp), ('ridge', ridge), ('linreg', linreg)],
    final_estimator=RidgeCV(store_cv_results=True)  # Sửa ở đây
)

# Huấn luyện mô hình Stacking
stacking_model.fit(scaled_X_train, y_train)

# Dự đoán trên tập train, validation và test
y_train_pred = stacking_model.predict(scaled_X_train)
y_val_pred = stacking_model.predict(scaled_X_val)
y_test_pred = stacking_model.predict(scaled_X_test)

# Đánh giá mô hình
print('R2 score on train set:', r2_score(y_train, y_train_pred))
print('R2 score on validation set:', r2_score(y_val, y_val_pred))
print('R2 score on test set:', r2_score(y_test, y_test_pred))

print('Mean Squared Error on train set:', mean_squared_error(y_train, y_train_pred))
print('Mean Squared Error on validation set:', mean_squared_error(y_val, y_val_pred))
print('Mean Squared Error on test set:', mean_squared_error(y_test, y_test_pred))
