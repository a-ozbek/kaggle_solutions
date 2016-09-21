import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.utils.np_utils import to_categorical

# Load the data
import pickle
X_train = pickle.load(open('X_train.pkl','rb'))
y_train = pickle.load(open('y_train.pkl', 'rb'))
X_train = X_train.reshape(42000,28,28,1)

# Define the model
model = Sequential()
model.add(Convolution2D(6,5,5, border_mode='same', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(16,5,5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# Flatten
model.add(Flatten())
# Fully Connected Layer
model.add(Dense(output_dim=120, activation='relu'))
model.add(Dense(output_dim=84, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))


# Define the optimizer and the loss function
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Fit the model
y_train = to_categorical(y_train)
model_hist = model.fit(X_train, y_train, batch_size=128, nb_epoch=12)




