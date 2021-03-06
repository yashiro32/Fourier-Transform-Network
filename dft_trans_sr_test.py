from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Reshape
from keras.utils import to_categorical

from keras.layers import Conv1D, GlobalMaxPooling1D

from dft_transform import DftTransform
#from stft import STFT

# Second dimension of the feature is dim2
feature_dim_2 = 11

# Save data to array file first
#save_data_to_array(max_len=feature_dim_2)

sampling_rate = 400
# # Loading train set and test set
#X_train, X_test, y_train, y_test = get_train_test()
X_train, X_test, y_train, y_test = get_traintest_raw()

# # Feature dimension
feature_dim_1 = 20
channel = 1
epochs = 200
batch_size = 10
verbose = 1
num_classes = 3

filters = 32
kernel_size = 3

# Reshaping to perform 2D convolution
#X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
#X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)
#X_train = X_train.reshape(X_train.shape[0], sampling_rate, channel)
#X_test = X_test.reshape(X_test.shape[0], sampling_rate, channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)


def get_model():
    model = Sequential()
    
    model.add(DftTransform(512, input_shape=(sampling_rate,)))
    #model.add(STFT(512, 312, 128, input_shape=(sampling_rate,)))
    #model.add(Reshape((sampling_rate, 1)))
    # we add a Convolution1D, which will learn filters word group filters of size filter_length
    #model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1, input_shape=(sampling_rate, channel)))
    #model.add(Conv1D(32, 2, padding='valid', activation='relu', strides=1, input_shape=(sampling_rate, channel)))
    #model.add(Conv1D(48, 2, padding='valid', activation='relu', strides=1))
    #model.add(Conv1D(120, 2, padding='valid', activation='relu', strides=1))
    #model.add(Flatten())
    # we use max pooling
    #model.add(GlobalMaxPooling1D())
    

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return model

# Predicts one sample
def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][np.argmax(model.predict(sample_reshaped))]

model = get_model()
model.summary()
model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))