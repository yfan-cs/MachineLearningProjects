""" Network for classifying the data in Fashion MNIST dataset """


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
from keras import backend as K

class Net2:

	def __init__(self):
		# train and test data:
		self.X_train_ = np.array([])
		self.y_train_ = np.array([])
		self.X_test_ = np.array([])
		self.y_test_ = np.array([])
		# network model:
		self.model_ = Sequential()
		# load data:
		self.LoadData()
		# set the data format to channel first
		K.set_image_data_format('channels_first')

	def LoadData(self):
		""" Load the data and convert them to [0,1] as float32 """
		self.trainImages_ = np.load('./Fashion MNIST/trainImages.npy')
		self.testImages_ = np.load('./Fashion MNIST/testImages.npy')

		self.X_train_ = np.float32(self.trainImages_)
		self.y_train_ = np.load('./Fashion MNIST/trainLabels.npy')
		self.X_test_ = np.float32(self.testImages_)
		self.y_test_ = np.load('./Fashion MNIST/testLabels.npy')

		# scale the data to range[0,1]
		self.X_train_ = self.X_train_ / 255
		self.X_test_ = self.X_test_ / 255

		# input shape and output number of classes
		self.input_shape_ = self.X_train_[0].shape
		self.units_ = 10

	def ConstructNetwork(self):
		""" Build the network using Keras"""
		self.model_.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=self.input_shape_))
		self.model_.add(Conv2D(64, (3, 3), activation='relu'))
		self.model_.add(MaxPooling2D(pool_size=(2, 2)))
		self.model_.add(Dropout(0.5))
		self.model_.add(Flatten())
		# Flattens the input. Does not affect the batch size.
		self.model_.add(Dense(128, activation='relu'))
		self.model_.add(Dropout(0.5))
		# Dropout consists in randomly setting a fraction rate of input units to 0 
		# at each update during training time, which helps prevent overfitting.
		self.model_.add(Dense(self.units_, activation='softmax'))
		self.model_.compile(loss=keras.losses.categorical_crossentropy,
		              optimizer=keras.optimizers.Adadelta(),
		              metrics=['accuracy'])
	def Train(self):
		""" Train the network """
		history = History()

		self.model_.fit(self.X_train_, self.y_train_,
          batch_size=128,
          epochs=15,
          verbose=1,
          callbacks=[history],
          validation_data=(self.X_test_, self.y_test_))
		# after training, print out some weights and biases:
		for layer in self.model_.layers:
			weights = layer.get_weights() # list of numpy arrays
			print("current layer weights:")
			print(weights)

		# after training, see the first 10 misclassified images:
		y_predict = self.model_.predict_classes(self.X_test_) # (10000,)
		y_test = np.argmax(self.y_test_,axis=1)
		
		wrong_idices = np.not_equal(y_predict,y_test)
		wrong_idices = [i for i,e in enumerate(wrong_idices) if e]

		for idx in range(10):
			plt.imshow(self.testImages_[wrong_idices[idx]][0,:,:], interpolation='nearest') 
			plt.show()
		
		# after training, draw train and test accuracy v.s. epochs plot
		num_epochs = history.epochs
		epochs = np.linspace(1.0, num_epochs, num_epochs)

		fig, ax = plt.subplots()
		ax.plot(epochs, history.accs)
		ax.plot(epochs, history.val_accs)
		ax.legend(['train accuracy', 'test accuracy'])
		ax.set(xlabel='epochs', ylabel='accuracy',
			title='train and test accuracy over epochs')
		plt.show()


class History(keras.callbacks.Callback):
	""" a callback to store train and test accuracy at each epoch """
	# acc, val_acc
	def on_train_begin(self, logs={}):
		self.accs=[]
		self.val_accs=[]
		self.epochs=0
	def on_epoch_end(self,epoch,logs={}):
		self.accs.append(logs.get('acc'))	
		self.val_accs.append(logs.get('val_acc'))
		self.epochs += 1


net = Net2()
net.ConstructNetwork()
net.Train()




