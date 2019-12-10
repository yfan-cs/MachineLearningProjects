
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
	''' Build the dataset '''
	def __init__(self, X, y, transform=None):

		self.img = X
		self.label = y
		self.transform = transform # transform on images

	def __getitem__(self, idx):

		label, img = self.label[idx], self.img[idx]
		if self.transform:
			img = self.transform(img)
		return label, img

	def __len__(self):
		return len(self.label)


# network model
class FlowerNet(nn.Module):
	def __init__(self):
		super(FlowerNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(32),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(p=0.3))
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(p=0.3))
		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(p=0.6))
		self.fc1 = nn.Sequential(
			nn.Linear(128*4*4,256),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Dropout(p=0.6)
			)

		self.fc2 = nn.Linear(256, 5)   # 5 classes of flowers

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		out = self.fc2(out)
		return out

	def LoadData(self):
		""" Load the data and convert them to [0,1] as float32. 
			Then divide them into train / test """
		Images = np.load('./Flowers/flower_imgs.npy')
		Labels = np.load('./Flowers/flower_labels.npy')
		n = Labels.size
		indices = list(range(n))
		split = int(np.floor(0.1 * n))
		# np.random.seed(1234)   # fix the training and test datasets for bettering tuning the parameters.
		np.random.shuffle(indices)
		train_indices, test_indices = indices[split:], indices[:split]
		self.img_mean = np.mean(np.swapaxes(Images/255.0,0,1).reshape(3,-1),1)
		self.img_std = np.std(np.swapaxes(Images/255.0,0,1).reshape(3,-1),1)

		normalize = transforms.Normalize(mean=list(self.img_mean), std=list(self.img_std))

		TrainTrans = transforms.Compose([
			transforms.ToPILImage(),
        	transforms.RandomCrop(28),
        	transforms.Resize(32),
        	transforms.RandomHorizontalFlip(),
        	transforms.RandomRotation(10),
        	transforms.ToTensor(),
        	normalize,
			])
		TestTrans = transforms.Compose([
			transforms.ToPILImage(),
			transforms.ToTensor(),
			normalize,
			])
		train_dataset = MyDataset(Images[train_indices],Labels[train_indices],TrainTrans)
		test_dataset = MyDataset(Images[test_indices], Labels[test_indices], TestTrans)

		self.Train_dataloader = DataLoader(dataset=train_dataset, batch_size = 128, shuffle=True)
		self.Test_dataloader = DataLoader(dataset=test_dataset, batch_size = 128, shuffle=False)

def train():

	model = FlowerNet().to(device)
	model.LoadData()
	optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.001)
	scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=10)
	criterion = nn.CrossEntropyLoss()  # see the torch.nn.modules.loss document
	Train_epoch = 200
	train_acc = []
	test_acc = []
	for epoch in range(Train_epoch):
		correct = 0
		total = 0
		train_loss = 0
		for batch, (label, image) in enumerate(model.Train_dataloader):
			label, image = label.to(device), image.to(device)
			output = model(image)
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
		for label, image in model.Test_dataloader:
			image = image.to(device)
			label = label.to(device)
			outputs = model(image)
			predicted = torch.argmax(outputs, dim=1)
			total += label.size(0)
			correct += (predicted == label).sum().item()
		print('Test accuracy of the model on the test images:{}%'.format(100 * correct/total))
		test_acc.append(correct / total)
	# see the learned weights in each layer:
	params = list(model.parameters())
	conv1 = params[0]
	conv2 = params[4]
	conv3 = params[8]
	fc1 = params[12]
	fc2 = params[16]
	print("First conv layer:")
	print(conv1[0][0])
	print("Second conv layer:")
	print(conv2[0][0])
	print("Third conv layer:")
	print(conv3[0][0])
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

	yticks = np.linspace(0.3, 0.9, 11)

	ax.set_yticks(yticks)
	plt.show()

	return model

def test(model):
	with torch.no_grad():
		correct = 0
		total = 0
		misclassified = []
		for label, image in model.Test_dataloader:
			image = image.to(device)
			label = label.to(device)
			outputs = model(image)
			predicted = torch.argmax(outputs, dim=1)
			total += label.size(0)
			correct += (predicted == label).sum().item()
			misclassified.extend(image[predicted != label])
		print('Test accuracy of the model on the test images:{}%'.format(100 * correct/total))

		# plot first 10 misclassified images:
		for i in range(10):
			img = misclassified[i+1].cpu().numpy()
			img[0,:,:] = img[0,:,:] * model.img_std[0] + model.img_mean[0]
			img[1,:,:] = img[1,:,:] * model.img_std[1] + model.img_mean[1]
			img[2,:,:] = img[2,:,:] * model.img_std[2] + model.img_mean[2]
			plt.figure(i)
			plt.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')
			plt.show()

def main():
	model = train()
	test(model)

if __name__ == '__main__':
	main()



