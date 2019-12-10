
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt


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
class AdultNet(nn.Module):
	def __init__(self):
		super(AdultNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Linear(67, 128),
			nn.ReLU(),
			nn.BatchNorm1d(128),
			nn.Dropout(p=0.5))

		self.layer2 = nn.Linear(128, 2)   #  >=50k ?

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		return out

	def LoadData(self):
		""" Load the data and convert them to [0,1] as float32. 
			Then divide them into train / test """
		X = np.load('./Adult/data.npy').astype('float32') / 255.0
		Labels = np.load('./Adult/labels.npy')
		n = Labels.size
		indices = list(range(n))
		split = int(np.floor(0.1 * n))
		# np.random.seed(1234)   # fix the training and test datasets for bettering tuning the parameters.
		np.random.shuffle(indices)
		train_indices, test_indices = indices[split:], indices[:split]

		
		train_dataset = MyDataset(X[train_indices],Labels[train_indices])
		test_dataset = MyDataset(X[test_indices], Labels[test_indices])

		self.Train_dataloader = DataLoader(dataset=train_dataset, batch_size = 256, shuffle=True)
		self.Test_dataloader = DataLoader(dataset=test_dataset, batch_size = 256, shuffle=False)

def train():

	model = AdultNet().to(device)
	model.LoadData()
	optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay = 1e-3)
	scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=10)
	criterion = nn.CrossEntropyLoss()  # see the torch.nn.modules.loss document
	Train_epoch = 20
	train_acc = []
	test_acc = []
	for epoch in range(Train_epoch):
		correct = 0
		total = 0
		train_loss = 0
		for batch, (label, x) in enumerate(model.Train_dataloader):
			label, x = label.to(device).long(), x.to(device)
			output = model(x)
			predicted = torch.argmax(output, dim=1)
			total += label.size(0)
			correct += (predicted == label).sum().item()
			loss = criterion(output, label)
			train_loss += loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		scheduler.step(train_loss)

		print('Loss Sum:{:.4f} Epoch[{}/{}]'.format(train_loss, epoch+1, Train_epoch))
		print('Acc:{:.4f}% Epoch[{}/{}]'.format(100 * correct / total, epoch+1, Train_epoch))
		train_acc.append(correct / total)

		# see the validation accuracy
		correct = 0
		total = 0
		for label, x in model.Test_dataloader:
			x = x.to(device)
			label = label.to(device).long()
			outputs = model(x)
			predicted = torch.argmax(outputs, dim=1)
			total += label.size(0)
			correct += (predicted == label).sum().item()
		print('Test accuracy of the model on the test data:{}%'.format(100 * correct/total))
		test_acc.append(correct / total)


	# see the learned weights in each layer:
	params = list(model.parameters())
	fc1 = params[0]
	fc2 = params[4]
	print("FC1:")
	print(fc1[0][:20])
	print("FC2:")
	print(fc2[0][:20])

	# plot the train / validate acc
	epochs = np.linspace(1.0, Train_epoch, Train_epoch)

	fig, ax = plt.subplots(1)
	ax.plot(epochs, train_acc)
	ax.plot(epochs, test_acc)
	ax.legend(['train accuracy', 'test accuracy'])
	ax.set(xlabel='epochs', ylabel='accuracy',
		title='train and test accuracy over epochs')
	plt.show()

	return model

def test(model):
	with torch.no_grad():
		correct = 0
		total = 0

		for label, x in model.Test_dataloader:

			label, x = label.to(device).long(), x.to(device)
			outputs = model(x)
			predicted = torch.argmax(outputs, dim=1)
			total += label.size(0)
			correct += (predicted == label).sum().item()
		print('Test accuracy of the model on the test images:{}%'.format(100 * correct/total))


def main():
	model = train()
	test(model)

if __name__ == '__main__':
	main()
