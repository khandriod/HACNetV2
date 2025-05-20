"""
Configuration file for HACNetV2 training and model parameters
"""

# Model parameters
MODEL_CONFIG = {
    'channels': 32,  # Number of channels in the first layer
    'device': 'cuda'  # Device to run the model on
}

# Training parameters
TRAIN_CONFIG = {
    'batch_size': 8,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'save_interval': 5,  # Save model every N epochs
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs'
}

# Dataset parameters
DATASET_CONFIG = {
    'train_data_dir': 'data/train',
    'val_data_dir': 'data/val',
    'test_data_dir': 'data/test',
    'image_size': (512, 512),  # (height, width)
    'num_workers': 4
}

# Loss function weights
LOSS_WEIGHTS = {
    'branch1': 0.25,
    'branch2': 0.25,
    'branch3': 0.25,
    'final': 0.25
}

# Augmentation parameters
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation_range': 30,
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2)
} 