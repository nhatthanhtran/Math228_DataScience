
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from estimators import CustomRegressors
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# f = lambda x: x[0] + x[1]
# g = lambda x: (x[0],x[0]+x[1])
# h = lambda x: (x**x, x + 1)
# print(f((1,1)))
# print(g((1,1)))
# print(h(np.array([1,2])))

# print(np.array(h(np.array([[1],[2]]))))



# t=2*np.random.rand(100,1) # unif. distributed on [0,2]
# t = np.sort(t, axis=0)
# y=4+3*t+np.random.randn(100,1);

# g = lambda x: np.array([x**0, x, x**2, x**3, np.exp(x)])

data = np.loadtxt("CaCovidInfMarch24toMidJuly.txt")
t = np.arange(1,len(data)+1,1)

X_train, X_test, y_train, y_test = train_test_split(t, data, train_size=90, shuffle=False)

def f(x): return np.array([x**0, x,x**2,x**3, np.exp(x)])

gd_params = {
    "lr": 0.001,
    "num_epochs": 100,
    "batch_size": 20
}

linear_reg = CustomRegressors(f, type='sgd', gd_params=gd_params)
linear_reg.fit(X_train,y_train)
print(linear_reg.arr_params)
y_pred = linear_reg.pred(X_train)
y_test_pred = linear_reg.pred(X_test)
plt.plot(X_train, y_train, "b.")
plt.plot(X_train,y_pred, "r-")
plt.plot(X_test, y_test, "b.")
plt.plot(X_test,y_test_pred, "b-")
plt.xlabel("time", fontsize=18)
plt.ylabel("y", fontsize=18)
mse_train = np.sqrt(mean_squared_error(y_train,y_pred))
mse_pred = np.sqrt(mean_squared_error(y_test, y_test_pred))
plt.title(f"mse_train: {mse_train} mse_pred: {mse_pred}")
plt.show()


# l = list(range(10))
# d = {str(a): ord(str(a)) for a in l}
# def dummy(*args, lr, ne, algo):
#     print(*args)
#     print(*kwargs)
#     print(kwargs)


# dummy(*l, **d)


# data = np.array([[0], [1], [2], [3]],dtype=float)


# print(X_train.shape)
# print(X_train.reshape(-1,1).shape)
# print(X_train.reshape(-1,1))
# print(data)
# print(data.shape)
# data2 = np.array([[1], [2], [3], [4]],dtype=float)
# scaler = StandardScaler()

# scaler.fit(data,data2)
# print(data,data2)
# print("mean,var,std, scale")
# print(scaler.mean_)
# print(scaler.var_)
# print(np.sqrt(scaler.var_))
# print(scaler.scale_)
# print("exact transform")
# print(np.divide((data - scaler.mean_), np.sqrt(scaler.var_)))

# print("transform")
# print(scaler.transform(data))

# print("exact inverse")
# print(np.multiply(scaler.transform(data), np.sqrt(scaler.var_)) + scaler.mean_)
# print("inverse transform")
# print(scaler.inverse_transform(scaler.transform(data)))

# data = data2
# print("exact transform")
# print(np.divide((data - scaler.mean_), np.sqrt(scaler.var_)))

# print("transform")
# print(scaler.transform(data))

# print("exact inverse")
# print(np.multiply(scaler.transform(data), np.sqrt(scaler.var_)) + scaler.mean_)
# print("inverse transform")
# print(scaler.inverse_transform(scaler.transform(data)))

