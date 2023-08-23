import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load and preprocess data
df = pd.read_csv("final.csv")
column_count = df.shape[1]
labels = df.iloc[:, -1]
data = df.drop(df.columns[column_count - 1], axis=1)

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dummy_y = keras.utils.to_categorical(encoded_Y, num_classes=5)

X_train, X_test, y_train, y_test = train_test_split(data, dummy_y, test_size=0.05, random_state=42, shuffle=True)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Learning rate scheduler
def lr_schedule(epoch):
    return 0.001 * (0.1 ** int(epoch / 10))

lr_scheduler = LearningRateScheduler(lr_schedule)

# Early stopping and model checkpoint
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Model definition
model = Sequential()
model.add(Dense(200, input_shape=(column_count - 1,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, callbacks=[lr_scheduler, early_stopping, checkpoint], validation_data=(X_test, y_test))

# Evaluate the model
score = model.evaluate(X_test, y_test, batch_size=100)
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))





