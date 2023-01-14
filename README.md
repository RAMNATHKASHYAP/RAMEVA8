# RAMEVA8

import torchvision # provide access to datasets, models, transforms, utils, etc
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import torch.optim as optim

....................................................................................................

This code imports several libraries that will be used in the program.

The first line imports the PyTorch library, which provides a set of tools for building and training neural networks.

The second line imports the torchvision library, which provides access to a variety of datasets, models, transforms, and utilities that can be used with PyTorch.

The third line imports a sublibrary from torchvision called transforms, which provides a set of pre-defined image transforms that can be applied to image datasets. These transforms can be used for data augmentation, normalization, and other pre-processing tasks.

The fourth line imports the random library, which provides functions for generating random numbers and selecting random elements from a list or array.

The fifth line imports the functional module of PyTorch's nn library, which provides a set of functions that can be used to define the forward pass of a neural network. It includes various activation functions, loss functions, and other functions commonly used in neural networks.

The sixth line imports optim module of Pytorch, which provides various optimization algorithms such as stochastic gradient descent, Adam, etc which can be used to optimize the parameters of a neural network.

.......................................................................................................

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self):
        # Load the MNIST dataset
        self.mnist_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform = transforms.Compose([
          transforms.ToTensor()
    ])
)

    def __len__(self):
        # Return the length of the MNIST dataset
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Get the image and label from the MNIST dataset
        image, label = self.mnist_dataset[idx]

        # Generate a random number between 0 and 9
        random_number = random.randint(0, 9)

        return image, random_number

# Create the dataset
dataset = MNISTDataset()

# Access an element from the dataset
image, random_number = dataset[5]
print(image)
print(random_number)
..........................................................................................................

This code creates a new class called MNISTDataset, which is a subclass of PyTorch's Dataset class. The Dataset class is a way of representing a collection of data in PyTorch, and the MNISTDataset class is a custom implementation of this class.

The class has three methods:

init(): In this method, it loads the FashionMNIST dataset using torchvision.datasets.FashionMNIST. It sets the root folder where the data should be downloaded, train=True, download=True to download the data if it does not exist, and applies the transform to change the data into tensors.

len(): This method returns the length of the MNIST dataset.

getitem(idx): This method returns the image and a random number between 0 and 9 at a specific index of the MNIST dataset.

After creating an instance of the class, it calls the getitem() method to access an element from the dataset and prints the image and random number.

It's worth mentioning that in this implementation the random number generated is not related to the label of the image, and this could be a problem if the label is needed for training or evaluation.
.......................................................................................................................................................

train_set = torchvision.datasets.MNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
...................................................................................................................

This code creates an instance of the torchvision.datasets.MNIST class, which is a dataset containing handwritten digits. The instance is created with the following arguments:

root: the root directory where the dataset should be downloaded. In this case, it is './data'.
train: set to True, so it loads the training set of the dataset.
download: set to True, so it downloads the dataset if it does not already exist in the root directory.
transform: a list of PyTorch transforms to apply to the data. In this case, it only applies the ToTensor() transform, which converts the images from PIL images to PyTorch tensors.
The result of this block of code is an object that represents the MNIST dataset, and it can be used to load and access the images and labels of the dataset.

........................................................................................................................................................

train_loader = torch.utils.data.DataLoader(
    train_set, 
    batch_size=32,
    shuffle=True
)
batch = next(iter(train_loader))
images, labels = batch
grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))
print('labels:', labels)

...........................................................................................................................................

This code creates a DataLoader object from the train_set created before. A DataLoader is a PyTorch utility that loads data from a dataset and provides convenient methods for iterating over the data in batches. The created DataLoader has the following arguments:

train_set: the dataset to load the data from.
batch_size: the number of samples in each batch. In this case, it is set to 32.
shuffle: set to True, so the order of the samples in the dataset is shuffled before each epoch.
The code also creates an iterator object by calling the iter() function on the train_loader and retrieves the first batch of data by calling the next() function on the iterator.
It then uses the torchvision.utils.make_grid() function to create a grid of images from the images in the batch. This function takes the images and the number of images in each row as arguments. The grid is then plotted using plt.imshow() function from matplotlib library and the labels are printed.

It should be noticed that this block of code will only work if the matplotlib library is imported and the images are being plotted.
..........................................................................................................................................................

import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, y):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x= F.relu(self.fc1(x))
        x = self.fc2(x)
      # Get the predicted number represented by the MNIST image
        _, predicted_num = x.max(dim=1)

        # Calculate the sum of the predicted number and the random number
        sum_output = predicted_num + y

        # One-hot encode the random number input
        rand_num_one_hot = torch.zeros(y.size(0), 10)
        rand_num_one_hot.scatter_(1, y.view(-1, 1), 1)

        # One-hot encode the sum output
        #sum_output_one_hot = torch.zeros(sum_output.size(0), 15)
        #sum_output_one_hot.scatter_(1, sum_output.view(-1, 1), 1)
        sum_output_one_hot = torch.zeros(sum_output.size(0), 15)
        sum_output_one_hot.scatter_(1, sum_output.view(-1, 1) % 15, 1)

        return x, rand_num_one_hot, sum_output_one_hot
...............................................................................................................
This code defines a new PyTorch neural network class called MNISTNet, which is a subclass of nn.Module.

The class has an init method that initializes several layers of the network:

conv1 is a 2D convolutional layer with 1 input channel, 10 output channels, and a kernel size of 5x5.
conv2 is also a 2D convolutional layer with 10 input channels, 20 output channels, and a kernel size of 5x5.
fc1 is a fully-connected linear layer with 320 input features and 50 output features
fc2 is a fully-connected linear layer with 50 input features and 10 output features
The class also has a forward method that defines the forward pass of the network.

In the forward method, the input x passed through the conv1 and conv2 layers with relu activation and max pooling layer.
Then it reshapes the output of the convolutional layers and pass it through the fc1 and fc2 layers.
After that, it calculates the predicted number represented by the MNIST image by getting the maximum value of the output of the fc2 layer along the dimension 1.
Then it calculates the sum of the predicted number and the random number y passed as an input.
Next, it one-hot encodes the random number y, so it can be used as an input to the network.
Finally, it one-hot encodes the sum output, so it can be used as an output of the network.
It's worth mentioning that this implementation is not going to work correctly as the sum of the predicted number and the random number is not in the range of [0,9], therefore the one hot encoding for the sum output will not be correct.
.................................................................................................................................

network = MNISTNet()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(10):
  total_loss = 0
  total_correct = 0
  for batch in train_loader: # Get Batch
    images, labels = batch 
    rand_num = torch.randint(0, 10, (1,))
    predictions, rand_num_one_hot, sum_output_one_hot = network(images, rand_num)
    loss = F.cross_entropy(predictions, labels) 
    loss.requires_grad_()

    optimizer.zero_grad()
    loss.backward() # Calculate Gradients
    optimizer.step() # Update Weights

    total_loss += loss.item()
    total_correct += get_num_correct(predictions, labels)

  print(
      "epoch:", epoch, 
      "total_correct:", total_correct, 
      "loss:", total_loss
    )

..........................................................................................................................................

This code creates an instance of the MNISTNet class defined earlier, and an instance of the Adam optimizer which is an optimization algorithm commonly used in deep learning.
It also creates a DataLoader from the train_set, with a batch_size of 100.

The code then enters a loop that iterates for 10 epochs. In each iteration, it initializes variables total_loss and total_correct to 0.

For each batch of data in the train_loader, it retrieves the images and labels, and generates a random number between 0 and 9 and pass them as input to the network. Then it gets the predictions, rand_num_one_hot and sum_output_one_hot from the forward pass.

It calculates the loss using cross-entropy loss function. Then, it sets the gradients of the network's parameters to zero, backpropagates the gradients through the network and updates the network's parameters using the optimizer's step() method.

It increments total_loss and total_correct by the loss value and the number of correctly classified images respectively.

At the end of each epoch, it prints the epoch number, total_correct and total_loss values.

It should be noticed that the code is missing the get_num_correct function which is used to calculate the total_correct variable.


........................................................................................................................................

