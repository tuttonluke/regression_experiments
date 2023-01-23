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
# %%
np.random.seed(42)
X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# splot test set into test and validation sets
X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                y_test, 
                                                                test_size=0.3)
sX = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler()

X_train_scaled = sX.fit_transform(X_train)
y_train_scaled = sy.fit_transform(y_train.reshape(-1, 1))

model = RandomForestRegressor()

parameter_distributions = {
    "max_samples" : uniform(0, 1),
    "n_estimators" : [2**n for n in range(1, 11)]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameter_distributions
)

random_search.fit(X_train_scaled, y_train_scaled)

print(random_search.best_params_)
print(random_search.best_score_)
#%% GRID SEARCH
np.random.seed(42)

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

grid = {
    "n_estimators" : [10, 50, 100],
    "criterion" : ["mse", "mae"],
    "min_samples_leaf" : [2, 1]
}

# for i, hyperparams in enumerate(grid_search(grid)):
#     print(i, hyperparams)

X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# splot test set into test and validation sets
X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                y_test, 
                                                                test_size=0.3)

best_hyperparams, best_loss = None, np.inf

for hyperparams in grid_search(grid):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_validation_pred = model.predict(X_validation)
    validation_loss = mean_squared_error(y_validation, y_validation_pred)

    print(f"H-Params: {hyperparams}\nValidation Loss: {validation_loss}")
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_hyperparams = hyperparams
    
print(f"Best Loss: {best_loss}\nBest Hyperparameters: {best_hyperparams}")
# %% K-FOLD
def k_fold(dataset, n_splits: int = 5):
    chunks = np.array_split(dataset, n_splits)
    for i in range(n_splits):
        training = chunks[:i] + chunks[i+1:]
        validation = chunks[i]
        yield np.concatenate(training), validation

loss = 0
n_splits = 5

for (X_train, X_validation), (y_train, y_validation) in zip(
    k_fold(X, n_splits), k_fold(y, n_splits)
):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_validation_pred = model.predict(X_validation)
    fold_loss = mean_squared_error(y_validation, y_validation_pred)
    loss += fold_loss
    print(f"Fold Loss: {fold_loss}")
print(f"K-Fold estimated loss: {loss / n_splits}")
# %% GRID SEARCH + K_FOLD
np.random.seed(42)

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

def k_fold(dataset, n_splits: int = 5):
    chunks = np.array_split(dataset, n_splits)
    for i in range(n_splits):
        training = chunks[:i] + chunks[i+1:]
        validation = chunks[i]
        yield np.concatenate(training), validation

X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# splot test set into test and validation sets
X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                y_test, 
                                                                test_size=0.3)

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