import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

mnist_train_data = torchvision.datasets.MNIST(download=True, root='MNIST/processed/training.pt',
                                              transform=transforms.ToTensor())
mnist_test_data = torchvision.datasets.MNIST(download=True, root='MNIST/processed/test.pt')

batch_size = 16
train_data_loader = torch.utils.data.DataLoader(dataset=mnist_train_data, batch_size=batch_size)
test_data_loader = torch.utils.data.DataLoader(dataset=mnist_test_data, batch_size=batch_size)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding="same")
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding="same")

        self.flatten = nn.Flatten()

        self.output_layer1 = nn.Sequential(
            nn.Linear(in_features=128*7*7, out_features=128),
            nn.ReLU())

        self.output_layer2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.max_pool(x)  # drop down to 14*14
        y = self.conv_layer2(x)
        x = self.max_pool(y)  # drop down to 7*7
        x = self.conv_layer3(x)

        x = self.flatten(x)
        x = self.output_layer1(x)
        x = nn.Dropout(p=0.5)(x)
        x = self.output_layer2(x)
        return x


model = NeuralNetwork()
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            break
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_data_loader, model, loss_fn, optimizer)
    test(test_data_loader, model, loss_fn)
print("Done!")
