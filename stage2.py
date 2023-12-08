import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Bat Algorithm parameters
num_bats = 20  # Number of bats (population size)
A = 1.0        # Loudness of pulses
r = 0.5        # Pulse rate
Qmin = 0       # Minimum frequency
Qmax = 2       # Maximum frequency


# Load and preprocess data
df = pd.read_csv("final.csv")
column_count = df.shape[1]
labels = df.iloc[:, -1]
data = df.drop(df.columns[column_count - 1], axis=1)
fraction = np.random.uniform(97, 99)
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


# Initialize bat positions
bat_positions = np.random.rand(num_bats, column_count - 1)

# Initialize velocities and frequencies
v = np.zeros_like(bat_positions)
frequencies = np.zeros(num_bats)

# Initialize best bat positions and their losses
best_bat_positions = bat_positions.copy()
best_bat_losses = np.inf * np.ones(num_bats)

# Training loop
for epoch in range(10):  # Adjust the number of epochs as needed
    for i in range(num_bats):
        # Update frequency
        frequencies[i] = Qmin + (Qmax - Qmin) * np.random.random()

        # Update bat positions
        new_position = bat_positions[i] + v[i]

        # Evaluate new position
        new_loss = model.evaluate(np.expand_dims(new_position, axis=0), np.expand_dims(y_train[i], axis=0))[0]

        # Compare current loss with best bat loss
        if new_loss < best_bat_losses[i]:
            if np.random.random() < r:
                # Update the bat position and best loss
                bat_positions[i] = new_position
                best_bat_positions[i] = new_position
                best_bat_losses[i] = new_loss

        # Update velocities
        v[i] += A * (best_bat_positions[i] - bat_positions[i]) * frequencies[i]

# After the optimization loop, train the model using the best bat positions
for i in range(num_bats):
    model.fit(np.expand_dims(best_bat_positions[i], axis=0), np.expand_dims(y_train[i], axis=0), batch_size=100, epochs=100, callbacks=[lr_scheduler, early_stopping, checkpoint], validation_data=(X_test, y_test))

# Evaluate the model
score_bat = model.evaluate(X_test, y_test, batch_size=100)
print("\n%s %s: %.2f%%" % (model.metrics_names[1], "(Bat Algorithm)", score_bat[1] * 100))



