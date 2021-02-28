from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import scale
from keras.losses import mse
from keras.activations import relu
from keras.regularizers import l2

from feature_engineer_api import FeatureEng, make_one_hot
from plot_history import plot_history

data_train_raw = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')
data_submistion_raw = pd.read_csv('./house-prices-advanced-regression-techniques/test.csv')

label = data_train_raw['SalePrice']
data = data_train_raw.drop(['SalePrice', 'Id'], axis=1)
id_submission = data_submistion_raw['Id']
X_submission = data_submistion_raw.drop('Id', axis=1)

data_plus_submission = pd.concat([data, X_submission], axis=0)
ins = FeatureEng(data_plus_submission)
objects_fields = ins.show_columns_are_ojebct()
for i in objects_fields:
    ins.fill_null(i)
data_plus_submission = make_one_hot(ins.data, objects_fields)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data_plus_submission = IterativeImputer().fit_transform(data_plus_submission)

data, X_submission = data_plus_submission[:len(data)], data_plus_submission[len(data):]
x_train, x_test, y_train, y_test = train_test_split(scale(data), label, shuffle=True, test_size=0.1)
rows, cols = data.shape
# ============================================
from keras.metrics import MeanSquaredError

model = Sequential()
model.add(Dense(300, input_dim=cols, activity_regularizer=l2(0.01)))
model.compile(optimizer=SGD(lr=0.1), loss='mse')
his = model.fit(x_train, y_train, validation_split=0.1, epochs=30)
plot_history(his)

loss_test = model.evaluate(x_test, y_test)
loss_train = his.history['loss'][-1]
loss_val = his.history['val_loss'][-1]

print(loss_train, loss_val, loss_test)

# ============================================
import pandas as pd

df = pd.DataFrame([loss_train, loss_val, loss_test], index=['train', 'val', 'test'])
# ============================================
