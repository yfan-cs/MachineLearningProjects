
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

# simple feed forward autoencoder, at least 1 layer with less than 16 neurons.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
	''' Build the dataset '''
	def __init__(self, X, y, transform=None):

		self.X = X
		self.label = y
		self.transform = transform # transform on inputs

	def __getitem__(self, idx):

		label, x = self.label[idx], self.X[idx]
		if self.transform:
			x = self.transform(x)
		return label, x

	def __len__(self):
		return len(self.label)


# network model
class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(33, 128),
			nn.Tanh(),
			nn.Linear(128,16),
			)

		self.decoder = nn.Sequential(
			nn.Linear(16,128),
			nn.Tanh(),
			nn.Linear(128,33),
			nn.Sigmoid(), # compress to a range(0,1)
			)


	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded

	def LoadData(self):
		""" Load the data and convert them to [0,1] as float32. 
			Then divide them into train / test """
		X = torch.Tensor(np.genfromtxt('./Three Meter/data.csv',delimiter=',').astype('float32'))
		min_vals, min_ids = torch.min(X, 1, keepdim=True)
		max_vals, max_ids = torch.max(X, 1, keepdim=True)
		X = (X-min_vals) / (max_vals - min_vals)


		self.dataloader = DataLoader(dataset=X, batch_size = 256, shuffle=True)


def train():

	model = AutoEncoder().to(device)
	model.LoadData()
	optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay = 0)
	scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=10)

	loss_func = nn.L1Loss()  # default reduction = 'elementwise_mean'
	Train_epoch = 128
	Train_loss = []
	for epoch in range(Train_epoch):
		for batch, x in enumerate(model.dataloader):
			x = x.to(device)
			encoded, decoded = model(x)

			loss = loss_func(decoded, x)
			train_loss = loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if batch % 100 == 0:
				print('Loss:{:.4f} Epoch[{}/{}]'.format(train_loss, epoch+1, Train_epoch))
				print('loss in loss.data[0]: {}'.format(loss.data[0]))

		scheduler.step(train_loss)

		Train_loss.append(train_loss)

	# see the learned weights in each layer:
	params = list(model.parameters())
	fc1 = params[0]
	fc2 = params[2]
	fc3 = params[4]
	fc4 = params[6]
	print("FC1:")
	print(fc1[0][:10])
	print("FC2:")
	print(fc2[0][:10])
	print("FC3:")
	print(fc3[0][:10])
	print("FC4:")
	print(fc4[0][:10])

	# plot the loss
	epochs = np.linspace(1.0, Train_epoch, Train_epoch)

	fig, ax = plt.subplots(1)
	ax.plot(epochs, Train_loss)
	ax.set(xlabel='epochs', ylabel='loss',
		title='MAE loss over epochs')
	plt.show()

	return model

def main():
	model = train()

if __name__ == '__main__':
	main()