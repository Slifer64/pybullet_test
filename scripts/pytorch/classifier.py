import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions from 1 to end (i.e. except 0 dim, which is the batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(self.fc3(x))
        return x

    def predict(self, inputs):
        return torch.max(self(inputs), 1)[1]


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def trainModel(transform, batch_size, classes, model_path):

    # ============  1. Load and normalize CIFAR10  ============
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # # get some random training images
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    #
    # # print labels
    # print("Labels: ", *['{0:10s}'.format(classes[labels[j]]) for j in range(batch_size)], sep='')
    #
    # # show images
    # imshow(torchvision.utils.make_grid(images))

    # ============  2. Define a Convolutional Neural Network  ============
    net = Net()

    # ============  3. Define a Loss function and optimizer  ============
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # ============  4. Train the network  ============
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                # print('[{0:d}, {1:5d}] loss: {2:.3f}'.format(epoch + 1, i + 1, running_loss / 2000))
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # ============  Save the model  ============
    torch.save(net.state_dict(), model_path)


def testModel(transform, batch_size, classes, model_path):

    # ============  5. Test the network on the test data  ============

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net()
    net.load_state_dict(torch.load(model_path))

    data_iter = iter(test_loader)
    images, labels = data_iter.next()

    # # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', *['{0:5s}'.format(classes[labels[j]]) for j in range(batch_size)])
    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', *['{0:5s}'.format(classes[predicted[j]]) for j in range(batch_size)])

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # # calculate outputs by running images through the network
            # outputs = net(images)
            # # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(outputs.data, 1)
            predicted = net.predict(images)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images:', 100 * correct // total, '%')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # outputs = net(images)
            # _, predictions = torch.max(outputs, 1)
            predictions = net.predict(images)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                # if label == prediction:
                #     correct_pred[classes[label]] += 1
                correct_pred[classes[label]] += label == prediction
                total_pred[classes[label]] += 1

    # print accuracy for each class
    print('-' * (1 + 7 + 3 + 10 + 1))
    print('|%7s | %10s|' % ("class","accuracy"))
    print('-' * (1 + 7 + 3 + 10 + 1))
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print('|%7s | %8.1f %%|' % (classname, accuracy))



if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model_path = './data/cifar_net.pth'

    # trainModel(transform=transform, batch_size=4, classes=classes, model_path=model_path)
    testModel(transform=transform, batch_size=1000, classes=classes, model_path=model_path)



