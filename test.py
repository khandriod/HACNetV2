import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from models.hacnetv2 import HACNetV2
from utils.data_utils import get_data_loaders
from config.config import MODEL_CONFIG, DATASET_CONFIG

def save_prediction(image, mask, prediction, save_path):
    """Save the original image, ground truth mask, and prediction"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(image.squeeze().cpu().numpy().transpose(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(132)
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Prediction
    plt.subplot(133)
    plt.imshow(prediction.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()

def test(model, test_loader, device, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc='Testing')):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            predictions = torch.sigmoid(outputs[-1])  # Use final output
            
            # Save predictions
            for j in range(images.size(0)):
                save_path = os.path.join(save_dir, f'prediction_{i}_{j}.png')
                save_prediction(
                    images[j],
                    masks[j],
                    predictions[j],
                    save_path
                )

def main():
    parser = argparse.ArgumentParser(description='Test HACNetV2')
    parser.add_argument('--config', type=str, default='config/config.py',
                      help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default='predictions',
                      help='Directory to save predictions')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(MODEL_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = HACNetV2(channels=MODEL_CONFIG['channels']).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create data loader
    _, _, test_loader = get_data_loaders(DATASET_CONFIG)
    
    # Test model
    test(model, test_loader, device, args.save_dir)
    print(f"Predictions saved to {args.save_dir}")

if __name__ == '__main__':
    main() 