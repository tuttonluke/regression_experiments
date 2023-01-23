# %%
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

# %%
class GradientDescent:
    def __init__(self) -> None:
        pass

    def min_max_norm(self, X, y):
        sX = preprocessing.MinMaxScaler()
        sy = preprocessing.MinMaxScaler()

        scaled_X = sX.fit_transform(X)
        scaled_y = sy.fit_transform(y.reshape(-1, 1))

        return scaled_X, scaled_y
    
    def batch_gradient_descent(self, X, y_true, epochs=1000, learning_rate = 0.01):
        number_of_features = X.shape[1]
        W = np.ones(shape=(number_of_features))
        b = 0
        total_samples = X.shape[0]

        cost_list = []
        epoch_list = []

        for epoch in range(epochs):
            y_pred = np.dot(W, X.T) + b

            W_grad = -(2 / total_samples) * (X.T.dot(y_true - y_pred))
            b_grad = -(2 / total_samples) * np.sum(y_true - y_pred)

            W = W - learning_rate * W_grad
            b = b - learning_rate * b_grad

            cost = np.mean(np.square(y_true - y_pred)) # MSE

            # batch 
            if epoch % 10 == 0:
                cost_list.append(cost)
                epoch_list.append(epoch)
            
        print(f"Weights: {W}\nBias: {b}\nLoss: {cost}")
        self.plot_cost(epoch_list, cost_list)
        return W, b, cost, cost_list, epoch_list
        
    def plot_cost(self, epoch_list, cost_list):
        plt.xlabel("Epoch")
        plt.ylabel("MSE Cost")
        plt.plot(epoch_list, cost_list)
        plt.show()
           
# %%
np.random.seed(42)
X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# splot test set into test and validation sets
X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                y_test, 
                                                                test_size=0.3)


model = GradientDescent()
X_train_scaled, y_train_scaled = model.min_max_norm(X_train, y_train)    

W, b, cost, cost_list, epoch_list = model.batch_gradient_descent(
    X_train_scaled, y_train_scaled.reshape(y_train_scaled.shape[0],),  
)

# %%
