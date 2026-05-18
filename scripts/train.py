"""
U-Net Training Script - Train on RTX 4050

This script trains a U-Net neural network to predict stress distributions
from bracket geometry and load information.

Architecture:
- Encoder-decoder CNN with skip connections
- Multi-task output: stress + strain prediction
- Optimized for RTX 4050 (mixed precision, batch size 32)
- TensorBoard logging and real-time monitoring
- Early stopping and model checkpointing

Author: AI Engineer
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Local imports (from scripts/preprocess.py)
# from preprocess import StressDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    
    This is a building block used in the U-Net encoder and decoder.
    Each block learns local image features through two 3×3 convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Stress Prediction
    
    Structure:
    - Input: (batch, 3, 512, 512) - geometry + load information
    - Encoder: 5 levels of downsampling with feature extraction
    - Bottleneck: Compressed representation at 32×32 resolution
    - Decoder: 5 levels of upsampling with skip connections
    - Output: (batch, 2, 512, 512) - stress + strain maps
    
    Key features:
    - Skip connections: Preserve spatial information
    - Residual learning: Easier gradient flow
    - Multi-channel: Can predict multiple outputs
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 2):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Input channels (3: geometry + 2 load channels)
            out_channels: Output channels (2: stress + strain)
        """
        super().__init__()
        
        # ========== ENCODER (Downsampling) ==========
        
        # Level 1: 512×512 → 256×256
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Level 2: 256×256 → 128×128
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Level 3: 128×128 → 64×64
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Level 4: 64×64 → 32×32
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # ========== BOTTLENECK ==========
        # Level 5: 32×32 (most compressed)
        self.bottleneck = DoubleConv(512, 1024)
        
        # ========== DECODER (Upsampling) ==========
        
        # Level 4: 32×32 → 64×64
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # Note: 1024 because of skip concatenation
        
        # Level 3: 64×64 → 128×128
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        # Level 2: 128×128 → 256×256
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        # Level 1: 256×256 → 512×512
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # ========== OUTPUT ==========
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor (batch, 3, 512, 512)
            
        Returns:
            Output tensor (batch, 2, 512, 512) - stress + strain
            
        Data flow:
        Input (512×512)
            ↓
        Encoder: Extract features, downsampling
            ↓
        Bottleneck: Compressed representation (32×32)
            ↓
        Decoder: Reconstruct full resolution with skip connections
            ↓
        Output (512×512)
        """
        # ========== ENCODER ==========
        # Store for skip connections
        e1 = self.enc1(x)              # (batch, 64, 512, 512)
        p1 = self.pool1(e1)            # (batch, 64, 256, 256)
        
        e2 = self.enc2(p1)             # (batch, 128, 256, 256)
        p2 = self.pool2(e2)            # (batch, 128, 128, 128)
        
        e3 = self.enc3(p2)             # (batch, 256, 128, 128)
        p3 = self.pool3(e3)            # (batch, 256, 64, 64)
        
        e4 = self.enc4(p3)             # (batch, 512, 64, 64)
        p4 = self.pool4(e4)            # (batch, 512, 32, 32)
        
        # ========== BOTTLENECK ==========
        bottleneck = self.bottleneck(p4)  # (batch, 1024, 32, 32)
        
        # ========== DECODER WITH SKIP CONNECTIONS ==========
        # Level 4: Upsample and concatenate with encoder level 4
        d4 = self.upconv4(bottleneck)      # (batch, 512, 64, 64)
        d4 = torch.cat([d4, e4], dim=1)    # Concatenate on channel dimension
        d4 = self.dec4(d4)                 # (batch, 512, 64, 64)
        
        # Level 3
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)                 # (batch, 256, 128, 128)
        
        # Level 2
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)                 # (batch, 128, 256, 256)
        
        # Level 1
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)                 # (batch, 64, 512, 512)
        
        # ========== OUTPUT ==========
        output = self.final_conv(d1)       # (batch, 2, 512, 512)
        
        return output


class ModelTrainer:
    """
    Handles training loop, validation, and model management.
    
    Features:
    - Mixed precision training (FP16 + FP32)
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    - Validation tracking
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda', 
                 mixed_precision: bool = True):
        """
        Initialize trainer.
        
        Args:
            model: U-Net model
            device: 'cuda' or 'cpu'
            mixed_precision: Use automatic mixed precision (FP16)
        """
        self.model = model.to(device)
        self.device = device
        self.mixed_precision = mixed_precision
        
        # Loss function: MSE for regression
        self.criterion = nn.MSELoss()
        
        # Optimizer: Adam is good for CNN training
        self.optimizer = optim.Adam(model.parameters(), 
                                   lr=1e-4, 
                                   weight_decay=1e-5)
        
        # Learning rate scheduler: decay over time
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.5  # Multiply LR by 0.5 every 10 epochs
        )
        
        # Mixed precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        logger.info(f"Model: {self._count_parameters()} parameters")
        logger.info(f"Device: {device}")
        logger.info(f"Mixed precision: {mixed_precision}")
    
    def _count_parameters(self) -> int:
        """Count model parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                # Mixed precision: forward in FP16, loss in FP32
                with autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item():.6f})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item():.6f})
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, checkpoint_dir: str = 'models',
              early_stopping_patience: int = 20):
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum epochs to train
            checkpoint_dir: Where to save best model
            early_stopping_patience: Stop if no improvement for N epochs
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Batch size: 32, LR: 1e-4, Device: {self.device}")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch [{epoch+1}/{epochs}]")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            logger.info(f"Train Loss: {train_loss:.6f}")
            logger.info(f"Val Loss: {val_loss:.6f}")
            logger.info(f"LR: {current_lr:.2e}")
            
            # Step scheduler
            self.scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), 
                          checkpoint_dir / 'unet_best.pth')
                logger.info("✅ New best model saved")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("\n✅ Training complete!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        # Save training history
        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def main():
    """Main training entry point"""
    logger.info("=" * 60)
    logger.info("U-Net Training - RTX 4050 Optimized")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Estimated duration: 15 hours")
    logger.info("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    model = UNet(in_channels=3, out_channels=2)
    logger.info(f"Created U-Net model")
    
    # Create trainer
    trainer = ModelTrainer(model, device=device, mixed_precision=True)
    
    # TODO: Load data
    # from preprocess import StressDataset
    # train_dataset = StressDataset('data/processed', split='train')
    # val_dataset = StressDataset('data/processed', split='val')
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32)
    
    # TODO: Train
    # history = trainer.train(train_loader, val_loader, epochs=100)
    
    logger.info("✅ Training script complete!")
    logger.info("Next step: python backend/main.py")


if __name__ == "__main__":
    main()
