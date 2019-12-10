""" Network for classifying the data in Breast Cancer dataset """

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from keras.utils import plot_model

class Net3:

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

	def LoadData(self):
		""" Load the data, split them into training and test datasets,
		 and convert them to [0,1] as float32 """
		X = np.genfromtxt('./Breast Cancer/breastCancerData.csv',delimiter=',')
		y = np.genfromtxt('./Breast Cancer/breastCancerLabels.csv',delimiter=',')
		X = X.astype('float32')
		X = X / 10
		n = X.shape[0]
		num_train = (int)(n*4.0/5.0)  # 4/5 of the data is divided into training set
		self.X_train_ = X[:num_train]
		self.y_train_ = keras.utils.to_categorical(y[:num_train])
		self.X_test_ = X[num_train:]
		self.y_test_ = keras.utils.to_categorical(y[num_train:])

		# input shape and output number of classes
		self.input_shape_ = self.X_train_[0].shape
		self.units_ = 2    # benign or malignant

	def ConstructNetwork(self):
		""" Build the network using Keras"""
		self.model_.add(Dense(32,
                 activation='relu',
                 input_shape=self.input_shape_))
		self.model_.add(Dropout(0.25))
		self.model_.add(Dense(128, activation='relu'))
		self.model_.add(Dropout(0.5))
		self.model_.add(Dense(self.units_, activation='softmax'))
		self.model_.compile(loss=keras.losses.categorical_crossentropy,
		              optimizer=keras.optimizers.Adadelta(lr=0.5),
		              metrics=['accuracy'])
	def Train(self):
		""" Train the network """
		history = History()

		self.model_.fit(self.X_train_, self.y_train_,
          batch_size=64,
          epochs=15,
          verbose=1,
          callbacks=[history],
          validation_data=(self.X_test_, self.y_test_))
		# after training, print out some weights and biases:
		for layer in self.model_.layers:
			weights = layer.get_weights() # list of numpy arrays
			print("current layer weights:")
			print(weights)

		# after training, see the first 10 misclassified data:
		y_predict = self.model_.predict_classes(self.X_test_)
		y_test = np.argmax(self.y_test_,axis=1)
		
		wrong_idices = np.not_equal(y_predict,y_test)
		wrong_idices = [i for i,e in enumerate(wrong_idices) if e]

		print("The first few misclassified data are: ")

		for idx in range(np.minimum(5, len(wrong_idices))):
			print(self.X_test_[wrong_idices[idx]])
			
		
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


net = Net3()
net.ConstructNetwork()
net.Train()




