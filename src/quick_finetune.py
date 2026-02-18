import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

# Add the parent directory to path so Python can find 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import works
from src.model2 import model, device

print("🚀 Starting quick fine-tuning...")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Use forward slashes
train_path = 'D:/Users/Lenova/Desktop/YukTha/dataset/train'

# Check if path exists
if not os.path.exists(train_path):
    print(f"❌ Path does not exist: {train_path}")
    print("Please check the path and try again.")
    sys.exit(1)

print(f"✅ Found dataset at: {train_path}")

# Use the SAME transforms as your original training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load your updated dataset
try:
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit(1)

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=0  # Set to 0 for Windows
)

print(f"✅ Loaded {len(train_dataset)} images")
real_count = len([x for x in train_dataset.targets if x == 1])
ai_count = len([x for x in train_dataset.targets if x == 0])
print(f"   Real images: {real_count}")
print(f"   AI images: {ai_count}")

# VERY IMPORTANT: Use LOW learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Train for just 3 epochs
model.train()
for epoch in range(3):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} - Avg Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Save your NEW model
models_dir = 'D:/Users/Lenova/Desktop/YukTha/models'
os.makedirs(models_dir, exist_ok=True)
new_model_path = os.path.join(models_dir, "mobilenet_food_model_v5.pth")
torch.save(model.state_dict(), new_model_path)
print(f"✅ New model saved to {new_model_path}")

print("\n🎉 Fine-tuning complete! Don't forget to update model path in model2.py")