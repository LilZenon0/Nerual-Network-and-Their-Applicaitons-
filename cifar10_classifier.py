
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_dir = "./data"
num_epochs = 5         
initial_lr = 0.05        
batch_size = 512         
num_workers = 4          
pin_memory = True if device.type == "cuda" else False

# CIFAR-10 
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# data
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

# validation
val_frac = 0.1
n_total = len(train_set)
n_val = int(val_frac * n_total)
n_train = n_total - n_val
train_subset, val_subset = torch.utils.data.random_split(train_set, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_loader  = DataLoader(test_set,     batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

#  MLP model
class SimpleMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.fc1 = nn.Linear(3*32*32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = SimpleMLP(num_classes=10).to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# -
def manual_sgd_step(model, lr):
    # parameter
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
           
            p.data -= lr * p.grad.data
    # zero grad
    model.zero_grad()

# evaluation funciton
def evaluate(loader, model):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return running_loss/total, correct/total

# training
print("Start training (manual SGD, no torch.optim)...")
best_val_acc = 0.0
best_state = None

for epoch in range(1, num_epochs+1):
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
    total = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()


        manual_sgd_step(model, initial_lr)

       
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_subset)
    epoch_acc = correct / total
    val_loss, val_acc = evaluate(val_loader, model)
    epoch_time = time.time() - epoch_start

    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = { 'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc }

    print(f"Epoch {epoch}/{num_epochs}  time={epoch_time:.1f}s  train_loss={epoch_loss:.4f} train_acc={epoch_acc*100:.2f}%  val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

print("Training finished. Best val acc: {:.2f}%".format(best_val_acc*100.0))


if best_state is not None:
    model.load_state_dict(best_state['model_state_dict'])
test_loss, test_acc = evaluate(test_loader, model)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc*100:.2f}%")
