
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from estimators import CustomRegressors
import math
from sklearn.model_selection import train_test_split

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
t = np.arange(1,len(data),1)

X_train, X_test, y_train, y_test = train_test_split(t, data, train_size=90, shuffle=False)

def f(x): return np.array([x**0, x])

gd_params = {
    "lr": 0.001,
    "num_epochs": 100,
    "batch_size": 20
}

linear_reg = CustomRegressors(f, type='sgd', gd_params=gd_params)
linear_reg.fit(X_train,y_train)
print(linear_reg.arr_params)
y_pred = linear_reg.pred(t)
plt.plot(X_train, y_train, "b.")
plt.plot(X_train,y_pred, "r-")
plt.xlabel("time", fontsize=18)
plt.ylabel("y", fontsize=18)
plt.show()


# l = list(range(10))
# d = {str(a): ord(str(a)) for a in l}
# def dummy(*args, lr, ne, algo):
#     print(*args)
#     print(*kwargs)
#     print(kwargs)


# dummy(*l, **d)