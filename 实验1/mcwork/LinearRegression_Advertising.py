# 导入相关的包
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 读取数据
path = '../Advertising.csv'
data = open(path)
# pandas读取
data = pd.read_csv(path)
x = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# 绘制散点图分析相关性
# plt.figure(figsize=(9,12))
# plt.subplot(311)
# plt.plot(data['TV'], y, 'ro')
# plt.title('TV')
# plt.grid()
# plt.subplot(312)
# plt.plot(data['Radio'], y, 'g^')
# plt.title('Radio')
# plt.grid()
# plt.subplot(313)
# plt.plot(data['Newspaper'], y, 'b*')
# plt.title('Newspaper')
# plt.grid()
# plt.tight_layout()
# plt.show()

# 构建线性回归模型
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1) # 对数据集进行划分，返回的是训练数据与预测数据
linreg = LinearRegression()
    # 线性回归拟合
model = linreg.fit(x_train, y_train)

# 模型的预测及评价
y_hat = linreg.predict(np.array(x_test))
mse = np.average((y_hat - np.array(y_test)) ** 2) # 方差
rmse = np.sqrt(mse) # 标准差
print(mse, rmse)

t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label='Test') # plt.plot(x, y, format_string, **kwargs)
# format_string：控制曲线的格式字符串，可选，由颜色字符、风格字符和标记字符组成
plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right') # 图例位置
plt.grid()
plt.show()
