import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Create the neural network with 3 layers
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=128)  # Hidden layer 1
        self.fc2 = nn.Linear(in_features=128, out_features=64)     # Hidden layer 2
        self.fc3 = nn.Linear(in_features=64, out_features=10)      # Output layer (10 classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax needed because CrossEntropyLoss applies it
        return x

model = MyNetwork()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)

# Training loop and evaluation loop
n_epochs = 5
train_losses = []
test_losses = []

for epoch in range(n_epochs):
    model.train()  # Set to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)  # Average loss over epoch
    train_losses.append(epoch_loss)
    
    # Evaluation on test set
    model.eval()  # Set to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)  # Average loss over epoch
    test_losses.append(test_loss)

    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Save the model
torch.save(model.state_dict(), 'mnist_model.pth')

# Plot the loss curves for both training and test sets
plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, n_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves for Training and Testing')
plt.legend()
plt.show()
