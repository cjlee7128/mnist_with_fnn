'''
python mnist_with_fnn.py --batch_size 16 --epochs 5 --lr 1e-4 --data_path data 
''' 
import torch 
import torchvision 
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import matplotlib.pyplot as plt 

import argparse

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='MNIST with FNN') 
    parser.add_argument('--batch_size', type=int, default=16, help='batch size (default: 16)') 
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs (default: 5)') 
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-4)') 
    parser.add_argument('--data_path', type=str, default='data', help='path to save MNIST data (default: data)') 

    args = parser.parse_args() 
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') 

    transform = transforms.Compose([transforms.ToTensor()]) 

    trainset = torchvision.datasets.MNIST(root=args.data_path, train=True, 
                                        download=True, transform=transform) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                            shuffle=True) 

    testset = torchvision.datasets.MNIST(root=args.data_path, train=False, 
                                        download=True, transform=transform) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, 
                                            shuffle=True) 

    classes = tuple([str(i) for i in range(10)]) 

    class Net(nn.Module): 
        def __init__(self): 
            super().__init__() 
            self.fc1 = nn.LazyLinear(256) 
            self.fc2 = nn.LazyLinear(64) 
            self.fc3 = nn.LazyLinear(10) 

        def forward(self, x): 
            x = torch.flatten(x, 1) # flatten all dimensions except batch 
            x = F.relu(self.fc1(x)) 
            x = F.relu(self.fc2(x)) 
            x = self.fc3(x) 
            return x 
        
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr=args.lr) 

    EPOCHS = args.epochs

    train_loss_values = [] 
    test_loss_values = [] 
    for epoch in range(EPOCHS): 
        
        net.train() 
        train_running_loss = 0. 
        for i, data in enumerate(trainloader, 0): 
            inputs, labels = data[0].to(device), data[1].to(device) 

            optimizer.zero_grad() 

            outputs = net(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 

            train_running_loss += loss.item() * inputs.size(0) 
        
        train_loss_values.append(train_running_loss / len(trainloader))

        net.eval() 
        test_running_loss = 0. 
        for i, data in enumerate(testloader, 0): 
            inputs, labels = data[0].to(device), data[1].to(device) 
            outputs = net(inputs) 
            loss = criterion(outputs, labels) 

            test_running_loss += loss.item() * inputs.size(0) 
        
        test_loss_values.append(test_running_loss / len(testloader)) 

    x = [i+1 for i in range(EPOCHS)] 
    plt.plot(x, train_loss_values, label='training loss') 
    plt.plot(x, test_loss_values, label='test loss') 
    plt.xticks(x) 
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend() 
    plt.show() 