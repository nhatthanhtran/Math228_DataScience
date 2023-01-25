
import numpy as np


class CustomRegressors():
    def __init__(self, obj_func_parts, type='normal', gd_params=None) -> None:
        self.obj_func_parts = obj_func_parts
        self.type = type
        self.arr_params = None
        self.bln_fitted =  False
        self.gd_params = gd_params
        self._set_num_param()
    def fit(self, x, y):

        y = y.reshape(-1)
        x = x.reshape(-1)
        data_size = len(y)
        if self.type == 'normal': # Solving the normal equation exactly    
            self.arr_params = self._normal_solver(x, y, data_size).T
        elif self.type == 'gd':
            self._grad_descent_solver(x, y, data_size)
        elif self.type == 'sgd':
            self._sto_grad_descent_solver(x, y, data_size)
        else:
            Exception(f"Unregcognize type of solver: {type}")

        self.bln_fitted = True

    def pred(self, x):
        if self.bln_fitted:
            x = x.reshape(-1)
            matA = np.array(self.obj_func_parts(x)).T
            return matA@self.arr_params

        else:
            print("Need to fit before predict.")
            return None

    def _normal_solver(self, x, y, data_size):
        matA = np.array(self.obj_func_parts(x)).T

        return np.linalg.inv(matA.T.dot(matA)).dot(matA.T).dot(y)

    def _grad_descent_solver(self, x, y, data_size):
        num_epochs = self.gd_params["num_epochs"]
        lr = self.gd_params["lr"]
        matA = np.array(self.obj_func_parts(x)).T
        y = y.reshape(len(y),1)

        self.arr_params = np.random.randn(self.num_param, 1)
        for ep in range(num_epochs):
            self.arr_params = self.arr_params - lr*matA.T@(matA@self.arr_params-y)

    def _sto_grad_descent_solver(self, x, y, data_size):
        num_epochs = self.gd_params["num_epochs"]
        lr = self.gd_params["lr"]
        batch_size = self.gd_params["batch_size"]

        matA = np.array(self.obj_func_parts(x)).T
        y = y.reshape(len(y),1)
        self.arr_params = np.random.randn(self.num_param, 1)
        for ep in range(num_epochs):
            shuffled_idx = np.random.permutation(data_size)
            matA_shuffled = matA[shuffled_idx]
            y_shuffled = y[shuffled_idx]
            for i in range(0, data_size, batch_size):
                matAi = matA_shuffled[i:i+batch_size]
                yi = y_shuffled[i:i+batch_size]
                grad = 2/batch_size*matAi.T@(matAi@self.arr_params - yi)
                self.arr_params = self.arr_params - lr*grad
    
    def _set_num_param(self):
        self.num_param = self.obj_func_parts(0).size

    

    
    

        