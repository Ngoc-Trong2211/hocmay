from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Tải dữ liệu cổ phiếu
stock_data = pd.read_csv('bmi_data.csv')

# Xóa các hàng bị thiếu
stock_data = stock_data.dropna()

features = stock_data[['Age', 'Height(Inches)', 'Weight(Pounds)']]
target = stock_data['BMI']

# Chia dữ liệu thành 70% train và 30% còn lại
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)

# Chia tập còn lại thành 15% test và 15% validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
scaled_X_val = scaler.transform(X_val)

# GridSearchCV parameters
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'max_iter': [1000, 3000, 5000]
}

# Khởi tạo MLP Regressor
mlp = MLPRegressor(random_state=1)

# GridSearchCV để tìm kiếm tham số tối ưu
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(scaled_X_train, y_train)

# In ra tham số tốt nhất
print("Best parameters found:", grid_search.best_params_)

# Huấn luyện lại mô hình với tham số tốt nhất
best_mlp = grid_search.best_estimator_

# Dự đoán
y_train_pred = best_mlp.predict(scaled_X_train)
y_val_pred = best_mlp.predict(scaled_X_val)
y_test_pred = best_mlp.predict(scaled_X_test)

# Đánh giá mô hình
print('R2 score on train set:', r2_score(y_train, y_train_pred))
print('R2 score on validation set:', r2_score(y_val, y_val_pred))
print('R2 score on test set:', r2_score(y_test, y_test_pred))

print('Mean Squared Error on train set:', mean_squared_error(y_train, y_train_pred))
print('Mean Squared Error on validation set:', mean_squared_error(y_val, y_val_pred))
print('Mean Squared Error on test set:', mean_squared_error(y_test, y_test_pred))