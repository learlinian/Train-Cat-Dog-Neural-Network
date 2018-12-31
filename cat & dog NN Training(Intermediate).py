import tensorflow as tf
import pickle
import time
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

vgg16_model = tf.keras.applications.vgg16.VGG16()
model = Sequential()
for layer in vgg16_model.layers[:-1]:   # append all layers except for the output layer from VGG16 into new model
    model.add(layer)
# model.summary()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(1, activation='sigmoid'))
# model.summary()

X_train = pickle.load(open('X_train.pickle', 'rb'))
y_train = pickle.load(open('y_train.pickle', 'rb'))
X_val = pickle.load(open('X_val.pickle', 'rb'))
y_val = pickle.load(open('y_val.pickle', 'rb'))
X_test = pickle.load(open('X_test.pickle', 'rb'))
y_test = pickle.load(open('y_test.pickle', 'rb'))

NAME = 'VGG16-' + str(int(time.time()))
callbacks = [TensorBoard(log_dir='log_VGG16/{}'.format(NAME)),
             ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4),
             EarlyStopping(monitor='val_acc', patience=7)]  # keep patience small here as training size is much larger

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val), epochs=100, callbacks=callbacks)

model.save('{}.model'.format(NAME))
test_loss, test_acc = model.evaluate(x=X_test, y=y_test)
print('{}:  testing accuracy = {}'.format(NAME, test_acc))
with open('record.txt', 'a+') as f:
    f.write('{}:  testing loss = {};  testing accuracy = {}'.format(NAME, test_loss, test_acc))
    f.write('\n')
