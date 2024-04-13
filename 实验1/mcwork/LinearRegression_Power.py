import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 读取数据
path = '../CCPP.csv'
data = pd.read_csv(path)
# print(data) # AT      V       AP     RH      PE
x = data[['AT', 'V', 'AP', 'RH']] # 样本特征
y = data[['PE']] # 样本输出

# 构建线性回归模型
# 划分训练集和数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
linreg = LinearRegression()
linreg.fit(x_train, y_train)
print(linreg.intercept_) # 截距 [460.05727267]
print(linreg.coef_) # 系数 [[-1.96865472 -0.2392946   0.0568509  -0.15861467]]
# PE=460.05727267−1.96865472∗AT−0.2392946∗V+0.0568509∗AP-0.15861467∗RH

# 模型评价
# 通常对于线性回归来讲，我么一般使用均方差（MSE，Mean Squared Error）或者均方根差（RMSE，Root Mean Squared Error）来评价模型的好坏
# y_pred = linreg.predict(np.array(x_test))
# print("MSE:", metrics.mean_squared_error(y_test, y_pred)) # MSE: 20.837191547220357
# print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # RMSE: 4.564777272465805
# 采样10折交叉验证法
predicted_10 = cross_val_predict(linreg, x, y, cv=10)
print("MSE:", metrics.mean_squared_error(y, predicted_10)) # MSE: 20.793672509857533
print("RMSE:", np.sqrt(metrics.mean_squared_error(y, predicted_10))) # RMSE: 4.560007950635343

# 画图查看结果
# plt.plot(t, y_test, 'r--', linewidth=2, label='Test')
# plt.plot(t, y_pred, 'g--', linewidth=2, label='Predict')
fig,ax = plt.subplots()
ax.scatter(y, predicted_10)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.legend(loc='upper right')
plt.grid()
plt.show()