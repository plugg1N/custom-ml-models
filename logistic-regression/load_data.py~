import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def sk_to_df(data) -> list:
    X_data = data.data
    X_columns = data.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data.target
    y = pd.Series(y_data, name='target')

    return x, y

X, y = sk_to_df(load_breast_cancer())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
