import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.model import CustomResNet50, CustomEfficientNet
from utils.data_preprocessing import CustomDataset, train_transforms, val_transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your data paths and labels here
train_image_paths = []  # Add paths to training images
train_labels = []       # Add corresponding labels
val_image_paths = []    # Add paths to validation images
val_labels = []         # Add corresponding labels

# Create datasets and dataloaders
train_dataset = CustomDataset(train_image_paths, train_labels, transform=train_transforms)
val_dataset = CustomDataset(val_image_paths, val_labels, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, optimizer, and scheduler
model = CustomResNet50()  # Or CustomEfficientNet()
model.to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)
early_stopping_patience = 10

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * images.size(0)
    return epoch_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            epoch_loss += loss.item() * images.size(0)
    return epoch_loss / len(loader.dataset)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(50):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss = validate_epoch(model, val_loader, criterion)
    
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print('Early stopping triggered')
            break
