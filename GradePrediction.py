import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dữ liệu: số giờ học và điểm
hours_studied = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
scores = np.array([50, 55, 65, 70, 75, 85, 90, 95, 100, 105])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(hours_studied, scores, test_size=0.2, random_state=42)

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán điểm dựa trên số giờ học
predictions = model.predict(X_test)

# Hiển thị kết quả
#for i in range(len(X_test)):
 #   print(f'Số giờ học: {X_test[i][0]}, Điểm dự đoán: {predictions[i]}')

# Vẽ đồ thị
plt.scatter(hours_studied, scores, color='blue', label='Dữ liệu')
plt.plot(X_train, model.predict(X_train), color='red', label='Đường hồi quy')
plt.xlabel('Số giờ học')
plt.ylabel('Điểm')
plt.title('Hồi quy tuyến tính: Điểm dự đoán dựa trên số giờ học')
plt.legend()
plt.show()
