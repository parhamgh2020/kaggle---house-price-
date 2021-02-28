from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from enum import Enum


class Optimizer(Enum):
    SGD = 'sgd'
    Momentum = 'momentum'
    RMSProp = 2


class FeatureEng(dict):
    def __init__(self, data):
        self.data = data

    def show_columns_with_null(self):
        lst, counter = [], 1
        for i in self.data.columns:
            if self.data[i].isna().sum():
                lst.append(i)
                print(f'{counter}. {i} - {self.data[i].dtype} : {self.data[i].isna().sum()} nulls')
                print('----------------------------------------')
                counter += 1
        print(f'total columns with null: {len(lst)}')
        return lst

    def show_columns_are_ojebct(self):
        lst, counter = [], 1
        for i in self.data.columns:
            if self.data[i].dtype == 'object':
                lst.append(i)
                print(f'{counter}.  {i} - {self.data[i].dtype} ')
                print(
                    f"     number of nulls: {self.data[i].isna().sum()} , number of elements: {len(self.data[i].value_counts())}")
                print('----------------------------------------')
                counter += 1
        print(f'tatol object columns : {len(lst)}')
        return lst

    def fill_null(self, field):
        if self.data[field].dtype == 'object':
            self.data[field].fillna('*', inplace=True)
        elif input(f"field type is {self.data[field].dtype}, do u want to fill with median?") == "yes":
            self.data[field].fillna(self.data[field].median(), inplace=True)
            print(f'median of {field} column is {self.data[field].median()}')
        else:
            return self.data

    # def make_one_hot(self, field):
    #     temp = pd.get_dummies(self.data[field], prefix=field)
    #     self.data.drop(field, axis=1, inplace=True, errors='ignore')
    #     return pd.concat([self.data, temp], axis=1)

    def labelencode(self, field):
        if self.data[field].dtype != 'object':
            print(f'type of this field is {self.data[field].dtype}')
            return self.data
        print(self.data[field].value_counts())
        print(f'number of titles : {len(self.data[field].value_counts())}')
        self.data[field] = LabelEncoder().fit_transform(self.data.Street) if input(
            'do you want to proceed?') == 'yes' else \
            self.data[field]
        return self.data

    def labelencode_ordering(self, field):
        pass

    def impute(self):
        self.data = IterativeImputer().fit_transform(self.data)
        return self.data


def make_one_hot(data, field):
    if isinstance(field, list):
        for i in field:
            temp = pd.get_dummies(data[i], prefix=i)
            data.drop(i, axis=1, inplace=True)
            data = pd.concat([data, temp], axis=1)
        return data
    else:
        temp = pd.get_dummies(data[field], prefix=field)
        data.drop(field, axis=1, inplace=True)
        return pd.concat([data, temp], axis=1)


def mae_percentage(y_test, predictions):
    y , p = None, None
    if type(y_test) != 'numpy':
        y = y_test.to_numpy()
    p = predictions.reshape(-1)
    sum_ = 0
    for i in range(len(y)):
        sum_ += abs(y[i] - predictions[i]) / y[i]
    print(sum_ / len(y), '%')
