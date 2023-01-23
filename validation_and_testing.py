# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# %%
X, y = datasets.fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# splot test set into test and validation sets

X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                y_test, 
                                                                test_size=0.3)
#%%                                                                
np.random.seed(42)

models = [
    DecisionTreeRegressor(splitter="random"),
    SVR(),
    LinearRegression()
]

for model in models:
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)

    train_loss = mean_squared_error(y_train, y_train_pred)
    validation_loss = mean_squared_error(y_validation, y_validation_pred)
    test_loss = mean_squared_error(y_test, y_test_pred)

    print(
        f"{model.__class__.__name__}: "
        f"Train Loss: {train_loss} | Validation Loss: {validation_loss} |"
        f"Test Loss: {test_loss}."
    )
#%% Train/Test data splits
X, y = datasets.fetch_california_housing(return_X_y=True)
np.random.seed(12)

train_score_list = []
test_score_list = []
validation_score_list = []

for i in range(1, 100):
    # print(i/100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                y_test, 
                                                                test_size=i/100)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_validation_pred = model.predict(X_validation)


    train_loss = mean_squared_error(y_train, y_train_pred)
    test_loss = mean_squared_error(y_test, y_test_pred)
    validation_loss = mean_squared_error(y_validation, y_validation_pred)


    train_score_list.append(train_loss)
    test_score_list.append(test_loss)
    validation_score_list.append(validation_loss) 

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.scatter(np.arange(99), train_score_list, c="r", label="Train Set")
ax.scatter(np.arange(99), test_score_list, c="b", label="Test Set", marker="x")
ax.scatter(np.arange(99), test_score_list, c="b", label="Validation Set", marker="s")
ax.legend()
plt.show()
#%% EVALUATING REAL PERFORMANCE ON HOUSE PRICING
X, y = datasets.fetch_california_housing(return_X_y=True)

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

    # splot test set into test and validation sets

    X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                    y_test, 
                                                                    test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    train_loss = mean_squared_error(y_train, y_train_pred)
    test_loss = mean_squared_error(y_test, y_test_pred)
    difference = train_loss - test_loss

    print(f"""Round {i}\nTrain Loss: {train_loss}\nTest Loss {test_loss}.\nDifference: {difference}.\n""")

# %%
