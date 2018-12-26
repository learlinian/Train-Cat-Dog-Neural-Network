import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K
import time
import pickle

dense_layers = [0, 1, 2]
conv_layers = [1, 2, 3]
layer_sizes = [32, 64, 128]

# load sorted training data
X_train = pickle.load(open('X_train.pickle', 'rb'))
y_train = pickle.load(open('y_train.pickle', 'rb'))

# validation data is optional if using validation_split instead of validation_data when fitting data into model
# X_val = pickle.load(open('X_val.pickle', 'rb'))
# y_val = pickle.load(open('y_val.pickle', 'rb'))

X_test = pickle.load(open('X_test.pickle', 'rb'))
y_test = pickle.load(open('y_test.pickle', 'rb'))

for conv_layer in conv_layers:
    for layer_size in layer_sizes:
            for dense_layer in dense_layers:
                with tf.Session() as sess:
                    # set the name for the NN model
                    NAME = '{}-conv-{}-nodes-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
                    # set up callbacks
                    callbacks = [TensorBoard(log_dir='log_temp/{}'.format(NAME)),
                                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5),
                                 EarlyStopping(monitor='val_acc', mode='min', patience=15)]

                    model = Sequential()    # Initialize model as a sequential model

                    # Feed training data into a convolutional layer first
                    model.add(Conv2D(layer_size, (5, 5), input_shape=X_train.shape[1:]))  # Specify input shape, otherwise model cannot be saved 
                    model.add(BatchNormalization())
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(Dropout(rate=0.25))

                    # convolutional layer loop
                    for i in range(conv_layer-1):
                        model.add(Conv2D(layer_size, (5, 5)))
                        model.add(BatchNormalization())
                        model.add(Activation('relu'))
                        model.add(MaxPooling2D(pool_size=(2, 2)))
                        model.add(Dropout(rate=0.25))

                    model.add(Flatten())    # Flatten 3D training data into 1D before sent into dense layer

                    # dense layer loop
                    for i in range(dense_layer):
                        model.add(Dense(layer_size))
                        model.add(BatchNormalization())
                        model.add(Activation('relu'))
                        model.add(Dropout(rate=0.25))

                    model.add(Dense(1))
                    # model.add(BatchNormalization())
                    model.add(Activation('sigmoid'))    # Use sigmoid activation function in output layer for binary training

                    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                    model.fit(X_train, y_train, batch_size=64, validation_split=0.3, epochs=300, callbacks=callbacks)
                    model.summary()
                    model.save('{}.model'.format(NAME))  # save model
                    test_loss, test_acc = model.evaluate(x=X_test, y=y_test)    # evaluate model with testing data

                    # store testing accuracy result into text file for easy reference
                    print('{}:  testing accuracy = {}'.format(NAME, test_acc))
                    with open('record.txt', 'a+') as f:
                        f.write('{}:  testing loss = {};  testing accuracy = {}'.format(NAME, test_loss, test_acc))
                        f.write('\n')
                    del model   # delete model in order to save memory
                K.clear_session()   # clear tf session in order to save memory
