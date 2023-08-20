import pandas as pd
import numpy as np
import keras

from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras import regularizers
from keras import initializers

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("final.csv")
df.head()

column_count = df.shape[1]
labels = df.iloc[:,-1]

data = df.drop(df.columns[column_count-1],axis=1)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = keras.utils.to_categorical(encoded_Y,num_classes=5) #np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(data, dummy_y, test_size=0.05, random_state=42,shuffle = True)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

seed = 7
np.random.seed(seed)



model = Sequential()
model.add(Dense(200, input_shape=(column_count-1, ), activation='relu',kernel_regularizer=regularizers.l2(1e-5),kernel_initializer=keras.initializers.glorot_normal(seed=seed),bias_initializer='zeros'))
model.add(Dropout(0.8))
model.add(Dense(5, activation='softmax'))
model.summary()

sgd = keras.optimizers.Adadelta()
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['acc'])

model.fit(X_train, y_train,
          epochs=10,  #default = 10
          batch_size=100,
         shuffle=True)
score = model.evaluate(X_test, y_test, batch_size=100)
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))



n_classes = 5

model = Sequential()
model.add(Conv1D(kernel_size = 5, strides=1,filters = 32, activation='relu',input_shape=(column_count-1,1)))
                    
print(model.input_shape)
print(model.output_shape)

model.add(MaxPooling1D(pool_size = (2), strides=(2)))
print(model.output_shape)

#model.add(Conv1D (kernel_size = 5, strides=1, filters = 32, activation='relu'))
#print(model.output_shape)

model.add(MaxPooling1D(pool_size = (2), strides=(2)))
print(model.output_shape)

model.add(Flatten())

print(model.output_shape)

model.add(Dense (1000, activation='relu'))
print(model.output_shape)

model.add(Dense(n_classes, activation = 'softmax'))#,activity_regularizer=keras.regularizers.l2()))
print(model.output_shape)

#model.compile( loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=[keras.metrics.categorical_accuracy])
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])



#19 August 2023
X1 = np.expand_dims(X_train, axis=2)
model.fit(X1, y_train, epochs=10, batch_size=100)
X2 = np.expand_dims(X_test, axis=2)
scores = model.evaluate(X2, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))