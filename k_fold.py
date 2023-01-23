# %%
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import uniform
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import preprocessing
import itertools
import typing
from sklearn.metrics import mean_squared_error
import multiprocessing as mp

# %%
def min_max_norm(X, y):
    sX = preprocessing.MinMaxScaler()
    sy = preprocessing.MinMaxScaler()

    scaled_X = sX.fit_transform(X)
    scaled_y = sy.fit_transform(y.reshape(-1, 1))

    return scaled_X, scaled_y

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

def k_fold(dataset, n_splits: int = 5):
    chunks = np.array_split(dataset, n_splits)
    for i in range(n_splits):
        training = chunks[:i] + chunks[i+1:]
        validation = chunks[i]
        yield np.concatenate(training), validation

def k_fold_plus_grid_search(X_train, X_validation, y_train, y_validation):
    grid = {
    "n_estimators" : [10, 50, 100],
    "criterion" : ["mse", "mae"],
    "min_samples_leaf" : [2, 1]
    }
    n_splits = 5
    best_hyperparams, best_loss = None, np.inf

    # grid search first
    for hyperparams in grid_search(grid):
        loss = 0
        # instead of validation we use K-fold
        for (X_train, X_validation), (y_train, y_validation) in zip(
            k_fold(X, n_splits), k_fold(y, n_splits)
        ):
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            y_validation_pred = model.predict(X_validation)
            fold_loss = mean_squared_error(y_validation, y_validation_pred)
            loss += fold_loss
        # Take the mean of all the folds as the final validation score
        total_loss = loss / n_splits
        print(f"H-Params: {hyperparams}\nLoss: {total_loss}")
        if total_loss < best_loss:
            best_loss = total_loss
            best_hyperparams = hyperparams

    # See the final results
    print(f"Best loss: {best_loss}")
    print(f"Best hyperparameters: {best_hyperparams}")
# %%
if __name__ == "__main__":
    np.random.seed(42)
    X, y = datasets.fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # splot test set into test and validation sets
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                    y_test, 
                                                                    test_size=0.3)
    X_train_scaled, y_train_scaled = min_max_norm(X_train, y_train)
    # p1 = mp.Process(target=)
    
    
    # k_fold_plus_grid_search(X_train, X_validation, y_train, y_validation)
# %%
grid = {
    "n_estimators" : [10, 50, 100],
    "criterion" : ["mse", "mae"],
    "min_samples_leaf" : [2, 1]
    }
#%%
import multiprocessing as mp

def square(x):
    return x ** 2

number_list = [i for i in range(10)]

processes_dict = {}
for x in range(1, len(number_list) + 1):
    processes_dict["p_%02d" % x] = mp.Process(target=square, args=number_list)

results_dict = {}
for key, value in sorted(processes_dict.items()):
    results_dict[key] = processes_dict[key].start()

print(results_dict)
