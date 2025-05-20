import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from models.hacnetv2 import HACNetV2
from utils.data_utils import get_data_loaders
from config.config import MODEL_CONFIG, TRAIN_CONFIG, DATASET_CONFIG, LOSS_WEIGHTS

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, masks in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss for each branch
        loss = 0
        for i, output in enumerate(outputs[:-1]):
            loss += LOSS_WEIGHTS[f'branch{i+1}'] * criterion(output, masks)
        loss += LOSS_WEIGHTS['final'] * criterion(outputs[-1], masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            loss = 0
            for i, output in enumerate(outputs[:-1]):
                loss += LOSS_WEIGHTS[f'branch{i+1}'] * criterion(output, masks)
            loss += LOSS_WEIGHTS['final'] * criterion(outputs[-1], masks)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description='Train HACNetV2')
    parser.add_argument('--config', type=str, default='config/config.py',
                      help='Path to config file')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'],
                      help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'],
                      help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=TRAIN_CONFIG['learning_rate'],
                      help='Learning rate for optimizer')
    args = parser.parse_args()
    
    # Update config with command line arguments
    TRAIN_CONFIG['num_epochs'] = args.epochs
    TRAIN_CONFIG['batch_size'] = args.batch_size
    TRAIN_CONFIG['learning_rate'] = args.learning_rate
    
    # Create directories
    os.makedirs(TRAIN_CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(TRAIN_CONFIG['log_dir'], exist_ok=True)
    
    # Set device
    device = torch.device(MODEL_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = HACNetV2(channels=MODEL_CONFIG['channels']).to(device)
    
    # Create data loaders
    config = {**DATASET_CONFIG, **TRAIN_CONFIG}  # Merge configs
    train_loader, val_loader, _ = get_data_loaders(config)
    
    # Create loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Create tensorboard writer
    writer = SummaryWriter(TRAIN_CONFIG['log_dir'])
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        print(f'\nEpoch {epoch+1}/{TRAIN_CONFIG["num_epochs"]}')
        print(f'Batch size: {TRAIN_CONFIG["batch_size"]}')
        print(f'Learning rate: {TRAIN_CONFIG["learning_rate"]}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % TRAIN_CONFIG['save_interval'] == 0:
            checkpoint_path = os.path.join(
                TRAIN_CONFIG['checkpoint_dir'],
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(
                TRAIN_CONFIG['checkpoint_dir'],
                'best_model.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
    
    writer.close()

if __name__ == '__main__':
    main() 