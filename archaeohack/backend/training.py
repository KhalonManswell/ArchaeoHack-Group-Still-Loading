import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm


# ==================== DATASET CLASS ====================
class HandwritingDataset(Dataset):
    def __init__(self, image_folder, csv_file):
        self.image_folder = image_folder
        if not self.image_folder.endswith('/'):
            self.image_folder += '/'

        # Read CSV file and create label mapping
        df = pd.read_csv(csv_file)
        
        # Create correct label mapping - using consecutive integer labels
        unique_gardiner_nums = sorted(df['gardiner_num'].unique())
        self.label_map = {gardiner_num: idx for idx, gardiner_num in enumerate(unique_gardiner_nums)}
        self.num_classes = len(self.label_map)
        
        print(f"Label mapping example: {list(self.label_map.items())[:5]}")

        # Data preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((140, 140)),
            transforms.ToTensor(),
        ])

        # Keep only image files
        all_files = os.listdir(image_folder)
        self.images = []

        for f in all_files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and not f.startswith('.'):
                gardiner_num = f.split('.')[0]
                if gardiner_num in self.label_map:
                    self.images.append(f)
                else:
                    print(f"Warning: {gardiner_num} not in label mapping")

        print(f"Found {len(self.images)} valid images")
        print(f"Dataset has {self.num_classes} classes")
        
        # Verify label range
        if len(self.images) > 0:
            labels = [self.label_map[img.split('.')[0]] for img in self.images]
            print(f"Label range: [{min(labels)}, {max(labels)}] (should be [0, {self.num_classes-1}])")

    def __getitem__(self, idx):
        try:
            image_file = self.images[idx]
            image_path = os.path.join(self.image_folder, image_file)
            image = Image.open(image_path)

            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Apply preprocessing
            if self.transform:
                image = self.transform(image)

            # Extract gardiner number from filename and map to numeric label
            gardiner_num = image_file.split('.')[0]
            target = self.label_map[gardiner_num]
            target = torch.tensor(target, dtype=torch.long)

            return image, target
            
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            # Return a default image and label
            return torch.zeros(3, 140, 140), torch.tensor(0, dtype=torch.long)

    def __len__(self):
        return len(self.images)


# ==================== CNN MODEL ====================
class HieroglyphCNN(nn.Module):
    def __init__(self, num_classes):
        super(HieroglyphCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 140x140 -> 70x70

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 70x70 -> 35x35

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 35x35 -> 17x17
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 17 * 17, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==================== TRAINING FUNCTION ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path):
    print(f"\nStarting training on device: {device}")

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': model.classifier[-1].out_features
            }, model_save_path)
            print(f"✅ Best model saved to: {model_save_path}")
            print(f"✅ Validation accuracy: {val_acc:.2f}%")

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")


# ==================== MAIN PROGRAM ====================
if __name__ == "__main__":
    # Configuration
    CSV_FILE = '/Users/katherine/Desktop/archaeohack-starterpack-main/data/gardiner_hieroglyphs.csv'
    IMAGE_FOLDER = '/Users/katherine/Desktop/archaeohack-starterpack-main/data/utf-pngs'
    
    # EXPLICIT SAVE LOCATION
    MODEL_SAVE_PATH = '/Users/katherine/PycharmProjects/PythonProject1/hieroglyph_model.pth'
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Verify file paths
    print("\n=== PATH VERIFICATION ===")
    print(f"Image folder: {IMAGE_FOLDER}")
    print(f"CSV file: {CSV_FILE}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Error: Image folder not found: {IMAGE_FOLDER}")
        exit(1)
        
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: CSV file not found: {CSV_FILE}")
        exit(1)
        
    print("✅ All file paths are valid")

    # Create save directory if it doesn't exist
    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Load dataset
    print("\nLoading dataset...")
    full_dataset = HandwritingDataset(IMAGE_FOLDER, CSV_FILE)
    
    if len(full_dataset) == 0:
        print("❌ Error: Dataset is empty")
        exit(1)

    num_classes = full_dataset.num_classes

    # Split dataset (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set: {train_size} images")
    print(f"Validation set: {val_size} images")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Verify data format
    print("\nChecking data format...")
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")  # Should be [batch_size, 3, 140, 140]
        print(f"Batch label shape: {labels.shape}")  # Should be [batch_size]
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Label range: [{labels.min()}, {labels.max()}]")
        break

    # Verify label range
    print("\nVerifying label range...")
    all_labels = []
    for images, labels in train_loader:
        all_labels.extend(labels.tolist())
        if len(all_labels) > 100:
            break

    print(f"Training label range: [{min(all_labels)}, {max(all_labels)}]")
    print(f"Model output classes: {num_classes}")

    if max(all_labels) >= num_classes:
        print(f"❌ Error: Label value {max(all_labels)} exceeds model output range [0, {num_classes-1}]")
        exit(1)
    else:
        print("✅ Label range is correct")

    # Create model
    print(f"\nCreating model with {num_classes} classes...")
    model = HieroglyphCNN(num_classes=num_classes).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, MODEL_SAVE_PATH)

    # Final verification
    if os.path.exists(MODEL_SAVE_PATH):
        file_size = os.path.getsize(MODEL_SAVE_PATH) / 1024 / 1024
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {MODEL_SAVE_PATH}")
        print(f"📊 File size: {file_size:.2f} MB")
        
        # Test loading the model
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
            test_model = HieroglyphCNN(num_classes=checkpoint['num_classes'])
            test_model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model can be loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"\n❌ Error: Model file was not created at {MODEL_SAVE_PATH}")
