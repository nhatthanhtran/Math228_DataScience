
import numpy as np


class CustomRegressors():
    def __init__(self, obj_func, num_param, grad_obj_func=None, type='normal', num_epochs=10) -> None:
        self.obj_func = obj_func
        self.num_param = num_param
        self.grad_obj_func = grad_obj_func
        self.type = type
        self.num_epochs = num_epochs

    def fit(self, x, y):

        y = y.reshape(-1)
        data_size = len(y)
        if type == 'normal': # Solving the normal equation exactly    
            pass
        elif type == 'gd':
            pass
        elif type == 'sgd':
            pass
        else:
            Exception(f"Unregcognize type of solver: {type}")
    
    def normal_solver(self, x, y, data_size):
        matA = np.c_[np.ones((data_size,1)), x]
        

    def grad_descent_solver(self):
        pass

    def sto_grad_descent_solver(self):
        pass

    

    
    

        