from keras.datasets import mnist
from keras.utils import to_categorical
from keras import regularizers
from keras import layers
from keras import models
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot  as plt
from keras.utils import plot_model

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def alexnet_model(img_shape, n_classes=10, weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', activation='relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

	# Layer 4
	alexnet.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

	# Layer 5
	alexnet.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(4096, activation='relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(4096, activation='relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(n_classes, activation='softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet

model = alexnet_model((28, 28, 1))


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# plot_model(model, show_shapes=True, to_file='/Users/ilkin_galoev/Documents/Diplom/Практика_1/AlexNet_practice.png')

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_size = 60000
test_size = 10000

train_images = train_images[:train_size]
train_labels = train_labels[:train_size]

test_images = test_images[:test_size]
test_labels = test_labels[:test_size]

train_images = train_images.reshape((train_size, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((test_size, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

x_validation = train_images[:test_size]
partial_train_images = train_images[test_size:]

y_validation = train_labels[:test_size]
partial_train_labels = train_labels[test_size:]

epochs_count = 5

history = model.fit(partial_train_images, partial_train_labels, epochs=epochs_count, batch_size=64, validation_data=(x_validation, y_validation))
# history = model.fit(train_images, train_labels, epochs=epochs_count, batch_size=64)
model.save('alexnet_mnist_validation.h5')

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, epochs_count+1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()

acc_values = history_dict['acc'] 
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss', test_loss)
print('Test acc', test_acc)
