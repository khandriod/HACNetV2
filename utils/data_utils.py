import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class CrackDataset(Dataset):
    """Dataset class for crack detection"""
    def __init__(self, data_dir, image_size=(512, 512), transform=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        
        # Get all image files
        self.image_files = [f for f in os.listdir(os.path.join(data_dir, 'images')) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, 'images', img_name)
        mask_path = os.path.join(self.data_dir, 'masks', img_name.replace('.jpg', '.png'))
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = transforms.Compose([
                transforms.Resize(self.image_size),  # Resize mask to match image size
                transforms.ToTensor()
            ])(mask)
        else:
            image = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])(image)
            mask = transforms.Compose([
                transforms.Resize(self.image_size),  # Resize mask to match image size
                transforms.ToTensor()
            ])(mask)
        
        return image, mask

def get_data_loaders(config):
    """Create data loaders for training, validation and testing"""
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),  # Fixed rotation range
        transforms.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CrackDataset(
        config['train_data_dir'],
        image_size=config['image_size'],
        transform=train_transform
    )
    
    val_dataset = CrackDataset(
        config['val_data_dir'],
        image_size=config['image_size'],
        transform=val_transform
    )
    
    test_dataset = CrackDataset(
        config['test_data_dir'],
        image_size=config['image_size'],
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 