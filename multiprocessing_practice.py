# %%
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor
import typing
import itertools
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split


def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

def k_fold(dataset, n_splits: int = 5):
    chunks = np.array_split(dataset, n_splits)
    for i in range(n_splits):
        training = chunks[:i] + chunks[i+1:]
        validation = chunks[i]
        yield np.concatenate(training), validation

def k_fold_plus_grid_search(X, y, X_train, X_validation, y_train, y_validation, grid):
    n_splits = 5
    best_hyperparams, best_loss = None, np.inf

    # grid search first
    for hyperparams in grid_search(grid):
        loss = 0
        # instead of validation we use K-fold
        for (X_train, X_validation), (y_train, y_validation) in zip(
            k_fold(X, n_splits), k_fold(y, n_splits)
        ):
            model = RandomForestRegressor(**hyperparams)
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

def multi_k_fold_plus_grid_search(X, y, X_train, X_validation, y_train, y_validation, hyperparams):
    loss = 0
    n_splits = 5
    for (X_train, X_validation), (y_train, y_validation) in zip(
            k_fold(X, n_splits), k_fold(y, n_splits)
    ):
        model = RandomForestRegressor(**hyperparams)
        model.fit(X_train, y_train)

        y_validation_pred = model.predict(X_validation)
        fold_loss = mean_squared_error(y_validation, y_validation_pred)
        loss += fold_loss
    # Take the mean of all the folds as the final validation score
    total_loss = loss / n_splits
    return total_loss


if __name__ == "__main__":
    grid = {
        "n_estimators" : [10, 50, 100],
        "criterion" : ["squared_error", "absolute_error"],
        "min_samples_leaf" : [2, 1]
        }
    np.random.seed(42)
    X, y = datasets.fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # splot test set into test and validation sets
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                    y_test, 
                                                                    test_size=0.3)
    hyperparams_list = []
    for hyperparams in grid_search(grid):
            hyperparams_list.append(hyperparams)
    best_hyperparams, best_loss = None, np.inf
    with Pool() as pool:
        loss_list = pool.starmap(multi_k_fold_plus_grid_search, [
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[0]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[1]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[2]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[3]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[4]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[5]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[6]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[7]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[8]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[9]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[10]),
            (X, y, X_train, X_validation, y_train, y_validation, hyperparams_list[11])
        ], chunksize=10000) 


