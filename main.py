import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# pip install pytorch
# pip install tqdm

# don't use conda install because of (BS) dependency requirements

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Hyperparameters
num_epochs = 4
batch_size = 64
learning_rate = 0.001

# 3. Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                          transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 4. Define the CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # FC Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. The Live Visualization Engine
class LiveVisualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Training Dashboard')

        self.ax_loss = self.fig.add_subplot(3, 1, 1)
        self.ax_weights = self.fig.add_subplot(3, 1, 2)
        
        self.loss_history = []
        
    def update(self, epoch, step, total_steps, loss_val, model, images, labels, preds):
        self.loss_history.append(loss_val)
        self.ax_loss.clear()
        self.ax_loss.plot(self.loss_history, color='blue', linewidth=2)
        self.ax_loss.set_title(f"Live Training Loss (Epoch {epoch+1})")
        self.ax_loss.set_ylabel("Error (Loss)")
        self.ax_loss.grid(True, alpha=0.3)
        
        weights = model.layer1[0].weight.data.cpu().numpy()
        w_min, w_max = weights.min(), weights.max()
        weights = (weights - w_min) / (w_max - w_min)
        
        grid_img = np.zeros((4*3 + 3, 8*3 + 7))
        self.ax_weights.clear()
        
        for i in range(32):
            row = i // 8
            col = i % 8

        stitched = np.hstack([weights[i, 0] for i in range(16)])
        self.ax_weights.imshow(stitched, cmap='viridis')
        self.ax_weights.set_title("Weights (Layer 1 Filter)")
        self.ax_weights.axis('off')

        for ax in self.fig.get_axes():
            if ax not in [self.ax_loss, self.ax_weights]:
                ax.remove()

        for idx in range(8):
            ax = self.fig.add_subplot(3, 8, 17 + idx)
            img = images[idx].cpu().numpy().squeeze() * 0.5 + 0.5
            ax.imshow(img, cmap='gray')

            pred_digit = preds[idx].item()
            true_digit = labels[idx].item()
            color = 'green' if pred_digit == true_digit else 'red'
            
            ax.set_title(f"P:{pred_digit} | T:{true_digit}", color=color, fontsize=10, fontweight='bold')
            ax.axis('off')

        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.001)


# 6. Training with Live Viz
print("Initializing Dashboard...")
visualizer = LiveVisualizer()

total_step = len(train_loader)

for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total=total_step, leave=False)
    
    for i, (images, labels) in loop:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())
        
        if i % 10 == 0:
            _, predicted = torch.max(outputs.data, 1)
            visualizer.update(epoch, i, total_step, loss.item(), model, images, labels, predicted)

print("Training Finished!")

# plt.ioff()

# Turn off interactive mode

plt.show()

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Final Accuracy: {100 * correct / total} %')
