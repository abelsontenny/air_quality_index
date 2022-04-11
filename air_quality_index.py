import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch

df=pd.read_csv('Real_Combine.csv')
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LeakyReLU,PReLU,ELU
# from keras.layers import Dropout

# Hyperparameters
# How many number of hidden layers?
# how many  number of neurons in hidden layer?
# learning rate?

def build_model(hp):
    model=keras.Sequential()
    for i in range(hp.Int('num_layers',2,20)): #initialize for loop for interating between range of layers
        model.add(layers.Dense(units=hp.Int('units_'+str(1), #itterate through all layers between 2 and 20. Dense function is to create nodes
                                           min_value=32,
                                           max_value=512,
                                           step=32),
                               activation='relu'))
        model.add(layers.Dense(1,activation='linear'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',[1e-2,1e-3,1e-4])),#choice, try  and choose any of the values for learning rate
            loss='mean_absolute_error',
            metrics=['mean_absolute_error'])
        return model
tuner=RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3, #5x3=15 differnt iterations in random search
    directory='project',
    project_name='Air_quality_index')
summary=tuner.search_space_summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

tuner.search(X_train, y_train, epochs=5, validation_data=(X_test,y_test))

tuner.results_summary()




# Normalize your outputs by quantile normalizing or z scoring. 
# To be rigorous, compute this transformation on the training 
# data, not on the entire dataset. For example, with quantile 
# normalization, if an example is in the 60th percentile of 
# the t]raining set, it gets a value of 0.6. (You can also
# shift the quantile normalized values down by 0.5 so that the
#  0th percentile is -0.5 and the 100th percentile is +0.5).

# Add regularization, either by increasing the dropout rate or 
# adding L1 and L2 penalties to the weights. L1 regularization 
# is analogous to feature selection, and since you said that 
# reducing the number of features to 5 gives good performance, 
# L1 may also.

# If these still don't help, reduce the size of your network.
# This is not always the best idea since it can harm performance,
# but in your case you have a large number of first-layer neurons
# (1024) relative to input features (35) so it may help.

# Increase the batch size from 32 to 128. 128 is fairly standard
# and could potentially increase the stability of the optimization.
 













