""" File containing methods for Titanic Prediction


"""

# pylint: disable=invalid-name, attribute-defined-outside-init

import os
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np

from numpy.random import seed
seed(42)

def get_mapping_for_categorical_column(colname):
    """ Helper to return mapping for categorical columns """
    if colname == "Pclass":
        return {1 : 1, 2 : 2, 3 : 3}
    elif colname == "Sex":
        return {"male" : 1, "female" : 2}
    elif colname == "Cabin":
        return {"CA" : 1, "CB" : 2, "CC" : 3, "CD" : 4, "CE" : 5, "CF" : 6, "CT" : 7}
    elif colname == "Embarked":
        return {"S" : 1, "C": 2, "Q": 3}
    elif colname == "Title":
        return {'Mr.' : 1, 'Mrs.' : 3, 'Miss.' : 4, 'Master.' : 5, 'Don.' : 6, 'Rev.' :7 ,
                'Dr.' : 8, 'Mme.' :9, 'Ms.': 10, 'Major.': 11, 'Lady.': 12, 'Sir.' : 13,
                'Mlle.': 14, 'Col.' :15, 'Capt.' : 16, 'Countess.' : 17, 'Jonkheer.' : 18,
                'Dona.' : 19}

class EncodeFeature(BaseEstimator, TransformerMixin):
    """
    Inputs - Titanic DF
    - Maps Categorical columns as per mapping described above
    - Converts Mapped columns to dummy cols
    - Drops columns that we don't want
    """
    def fit(self, X, y=None):
        """ Fit Model"""
        return self

    def transform(self, X, y=None):
        """
        Encode Features to Categorical variables

        """
        columns_with_na = ["Age", "Cabin", "Embarked", "Fare"]
        categorical_columns = ["Pclass", "Sex", "Cabin", "Embarked"]
        columns_to_drop = ["Ticket", "PassengerId"]

        self.categorical_columns = categorical_columns

        X = X.drop(columns = columns_to_drop)

        for coli in columns_with_na:
            X[coli+"_to_predict"] = 0
            X.loc[X[coli].isna(), coli+"_to_predict"] = 1

        X.loc[X["Cabin_to_predict"] == 0, "Cabin"] = X[X["Cabin_to_predict"] == 0]["Cabin"] \
                                                                .apply(lambda x: "C" + str(x)[0])

        ## Encode Age column
        X.loc[(X["Age_to_predict"] == 0) & (X["Age"] < 18), "Age"] = 0
        X.loc[(X["Age_to_predict"] == 0) & (X["Age"] >= 18), "Age"] = 1
        X.loc[X["Age_to_predict"] == 1, "Age"] = np.nan

        ## Encode Name column
        X["Title"] = X["Name"].apply(lambda x: re.search(r"\S+\.", x).group(0))
        X["Title"] = X["Title"].map(get_mapping_for_categorical_column("Title"))
        X[list(get_mapping_for_categorical_column("Title").keys())] = \
                                                     pd.get_dummies(X["Title"], columns = ["Title"])

        X = X.drop(columns = ["Title"])
        X["Name"] = X["Name"].apply(len)

        ## encode Pclass
        X["Pclass"] = X["Pclass"].map(get_mapping_for_categorical_column("Pclass"))
        X[list(get_mapping_for_categorical_column("Pclass").keys())] = \
                                                   pd.get_dummies(X["Pclass"], columns = ["Pclass"])
        X = X.drop(columns = ["Pclass"])

        ## encode Sex
        X["Sex"] = X["Sex"].map(get_mapping_for_categorical_column("Sex"))
        X[list(get_mapping_for_categorical_column("Sex").keys())] = \
                                                         pd.get_dummies(X["Sex"], columns = ["Sex"])
        X = X.drop(columns = ["Sex"])

        ## encode Cabin
        X.loc[(X["Cabin_to_predict"] == 0), "Cabin"] = \
             X[X["Cabin_to_predict"] == 0]["Cabin"].map(get_mapping_for_categorical_column("Cabin"))

        X[list(get_mapping_for_categorical_column("Cabin").keys())] = \
                                                     pd.get_dummies(X["Cabin"], columns = ["Cabin"])
        X = X.drop(columns = ["Cabin"])

        ## encode Embarked
        X.loc[(X["Embarked_to_predict"] == 0), "Embarked"] = \
                                                X[X["Embarked_to_predict"] == 0]["Embarked"] \
                                                .map(get_mapping_for_categorical_column("Embarked"))

        X[list(get_mapping_for_categorical_column("Embarked").keys())] = \
                                               pd.get_dummies(X["Embarked"], columns = ["Embarked"])
        X = X.drop(columns = ["Embarked"])

        ## encode Fare
        X.loc[X["Fare"] < 50, "FareTmp"] = "F1"
        X.loc[(50 <= X["Fare"]) & (X["Fare"] < 100), "FareTmp"] = "F2"
        X.loc[X["Fare"] >= 100, "FareTmp"] = "F3"
        X[["F1", "F2", "F3"]] = pd.get_dummies(X["FareTmp"], columns = ["FareTmp"])
        X = X.drop(columns = ["Fare", "FareTmp"])

        return X

class ImputeFare(BaseEstimator, TransformerMixin):
    """ Imputation for Fare """
    def _split_x_data(self, X):
        X = X.drop(columns = ["Age", "Age_to_predict",
                              "Cabin_to_predict",
                              "Embarked_to_predict",
                              "S", "C", "Q",
                              "CA", "CB", "CC", "CD", "CE", "CF", "CT"])

        X_train_all = X[X["Fare_to_predict"] == 0]
        X_test_all = X[X["Fare_to_predict"] == 1]

        y_train = X_train_all[["F1", "F2", "F3"]]
        X_train = X_train_all.drop(columns = ["F1", "F2", "F3", "Fare_to_predict"])

        y_test = X_test_all[["F1", "F2", "F3"]]
        X_test = X_test_all.drop(columns = ["F1", "F2", "F3", "Fare_to_predict"])

        return X_train, X_test, y_train, y_test

    def _scale_data(self, X):
        scaler = MinMaxScaler()

        sub_x = X[["Name", "SibSp", "Parch"]]
        sub_x = scaler.fit_transform(sub_x)

        X[["Name", "SibSp", "Parch"]] = sub_x

        return X

    def fit(self, X, y=None):
        X_train, X_test, y_train, y_test = self._split_x_data(X)

        self.X_test = X_test
        self.y_test = y_test

        X_train = self._scale_data(X_train)

        self.model = Sequential()
        self.model.add(Dense(X_train.shape[1]))
        self.model.add(Dense(10))
        self.model.add(Dense(3, activation="softmax"))
        self.model.compile(optimizer="adam", loss = "categorical_crossentropy",
                                                                            metrics = ["accuracy"])
        self.model.fit(X_train, y_train, epochs = 50)

        max_values = X[["F1", "F2", "F3"]].sum().to_dict()
        self.max_col = max(max_values, key=max_values.get)

        return self

    def transform(self, X, y=None):
        """ Transform Data """
        self.x_test = self._scale_data(self.X_test)

        X.loc[X["Fare_to_predict"] == 1, self.max_col] = 1
        return X

class ImputeEmbarked(BaseEstimator, TransformerMixin):
    """ Impute Embarked Data """
    def fit(self, X, y=None):
        """" Fit Model to determine Embarked data """
        max_values = X[["S", "C", "Q"]].sum().to_dict()
        self.max_col = max(max_values, key=max_values.get)
        return self

    def transform(self, X, y=None):
        """ Transform Data """
        X.loc[X["Embarked_to_predict"] == 1, self.max_col] = 1
        return X

class ImputeAge(BaseEstimator, TransformerMixin):
    """ Impute Age """
    def _split_x_data(self, X, y=None):

        X = X.drop(columns = [
                              "Cabin_to_predict",
                              "CA", "CB", "CC", "CD", "CE", "CF", "CT"])

        X_train_all = X[X["Age_to_predict"] == 0]
        X_test_all = X[X["Age_to_predict"] == 1]

        y_train = X_train_all["Age"]
        X_train = X_train_all.drop(columns = ["Age", "Age_to_predict"])

        y_test = X_test_all["Age"]
        X_test = X_test_all.drop(columns = ["Age", "Age_to_predict"])

        return X_train, X_test, y_train, y_test

    def _scale_data(self, X, y=None):
        scaler = StandardScaler()

        return scaler.fit_transform(X)

    def fit(self, X, y=None):
        """ Fit Model for Age """
        X_train, X_test, y_train, y_test = self._split_x_data(X)

        self.X_test = X_test
        self.y_test = y_test

        X_train = self._scale_data(X_train)

        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

        return self

    def transform(self, X, y=None):
        """ Transform data for AGe """
        self.x_test = self._scale_data(self.X_test)
        y_pred = self.model.predict(self.X_test)

        X.loc[X["Age_to_predict"] == 1, "Age"] = y_pred
        return X

class ImputeCabin(BaseEstimator, TransformerMixin):
    """" Imputation model for Cabin """
    def fit(self, X, y=None):
        """ Fit Model for Cabin """
        max_values = X[["CA", "CB", "CC", "CD", "CE", "CF", "CE", "CT"]].sum().to_dict()
        self.max_col = max(max_values, key=max_values.get)
        return self

    def transform(self, X, y=None):
        """ Transform Data"""
        X.loc[X["Cabin_to_predict"] == 1, self.max_col] = 1
        return X


# class RunFinalModel(BaseEstimator, TransformerMixin):
#     def scale_features(self, X):
#         pass

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         return X

class RunModel:
    """ Create Final Model """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def predict(self):
        """ Predict Data """
        self.X_train = self.X[~self.y.isna()]
        self.y_train = self.y[~self.y.isna()]

        self.X_test = self.X[self.y.isna()]
        self.y_test = self.y[self.y.isna()]

        self.X_train = self._scaler(self.X_train)
        self.X_test = self._scaler(self.X_test)

        self._fit_model()

        return self.model.predict(self.X_train), self.model.predict(self.X_test)

    def _scaler(self, X):
        scaler = StandardScaler()

        sub_x = X[["Name", "SibSp", "Parch"]]
        sub_x = scaler.fit_transform(sub_x)

        X[["Name", "SibSp", "Parch"]] = sub_x

        return X

    def _fit_model(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)


## Run Model ##

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
df_all = df_train.append(df_test)

pipe = Pipeline(
    steps=[
        ("encode_features", EncodeFeature()),
        ("impute_fare", ImputeFare()),
        ("impute_embarked", ImputeEmbarked()),
        ("impute_age", ImputeAge()),
        ("impute_cabin", ImputeCabin())
    ]
)

survived = df_all["Survived"]
df_all = df_all.drop(columns = ["Survived"])

transformed_df = pipe.fit_transform(X = df_all)
train_pred, test_pred = RunModel(transformed_df, survived).predict()



survived = survived[~survived.isna()]
print(accuracy_score(train_pred, survived))
print(f1_score(train_pred, survived))

final_df = pd.DataFrame({"PassengerId" : df_all[survived.isna()]["PassengerId"],
                         "Survived" : test_pred})
final_df.to_csv(os.path.join("prediction", "submission.csv"), index = False)
