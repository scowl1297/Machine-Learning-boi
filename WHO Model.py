import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow
import sklearn


##---------PRE PROCESSING------------##


##import data##
dataset = pandas.read_csv('life_expectancy.csv')


##determine data structure##
print(dataset.head())
print(dataset.describe())


##Seperate intial datasets##
labels = dataset.iloc[:,-1]
features = dataset.iloc[:, 0:-1]


##Change categorical columns to numerical ones##
features = pandas.get_dummies(dataset)


##Split the data into a training and testing set##
features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, test_size  = 0.3, random_state = 20)


##Normalize numerical features##
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns  = numerical_features.columns
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder = 'passthrough')

##Change raw numerical training features to normalized numerical features##
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)


##---------MODEL CONSTRUCTION------------##



##Create an instance of the model##
my_model = tensorflow.keras.models.Sequential()

##Input Layer##
input = InputLayer(input_shape = (features.shape[1], ))
my_model.add(input)

##Hidden Layers##
my_model.add(Dense(64, activation = 'relu'))


##Output Layer##
my_model.add(Dense(1))
my_model.summary()


##Optimizer##
opt = Adam(learning_rate = 0.01)
my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

print(len(features_train_scaled))
print(len(labels_train))
##---------FITTING AND EVALUATING------------##

##Fit the model to the data##
my_model.fit(features_train_scaled, labels_train, epochs = 40, batch_size = 1, verbose = 1)

## Define the mean squared error and mean average error##
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)
print(res_mse)
print(res_mae)






