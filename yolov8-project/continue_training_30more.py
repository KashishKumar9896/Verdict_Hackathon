"""
Continue Training Script - 30 MORE EPOCHS
==========================================
This script continues training from the best model of previous 20-epoch training
for an additional 30 epochs.
"""

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import os

# ===========================
# Configuration
# ===========================

# Model path - continue from 20-epoch training
PREVIOUS_MODEL = "runs/detect/continue_training_20epochs/weights/best.pt"

# Check if previous training exists
if not Path(PREVIOUS_MODEL).exists():
    print(f"❌ Error: Previous model not found at {PREVIOUS_MODEL}")
    print("Make sure the 20-epoch training completed successfully")
    exit(1)

# Dataset config
DATA_CONFIG = "config/config.yaml"

# Training parameters
ADDITIONAL_EPOCHS = 30  # 30 more epochs
BATCH_SIZE = 8
IMAGE_SIZE = 640

print("=" * 90)
print("🚀 CONTINUING TRAINING - 30 MORE EPOCHS FROM 20-EPOCH MODEL")
print("=" * 90)
print(f"✅ Found previous best model: {PREVIOUS_MODEL}")
print()
print("✅ Configuration:")
print(f"   Model:               {PREVIOUS_MODEL}")
print(f"   Config:              {DATA_CONFIG}")
print(f"   Additional Epochs:   {ADDITIONAL_EPOCHS}")
print(f"   Batch Size:          {BATCH_SIZE}")
print(f"   Image Size:          {IMAGE_SIZE}")
print()

# ===========================
# Load Model
# ===========================
print("⚡ Loading previous best model...")
print()

model = YOLO(PREVIOUS_MODEL)

print("✅ Model loaded successfully!")
print(f"   Classes: {model.names}")
print()

# ===========================
# Training Configuration
# ===========================
print("=" * 90)
print("🔥 RESUMING TRAINING - 30 MORE EPOCHS")
print("=" * 90)
print()
print("🎯 Training Settings:")
print("  ✓ Strong augmentation for better generalization")
print("  ✓ Mosaic augmentation enabled")
print("  ✓ AdamW optimizer with cosine LR")
print("  ✓ Lower learning rate for fine-tuning")
print("  ✓ Early stopping patience: 50")
print()
print("⏱️  Estimated time: ~3 hours")
print("=" * 90)

# ===========================
# Start Training
# ===========================
results = model.train(
    # Data
    data=DATA_CONFIG,
    
    # Training duration
    epochs=ADDITIONAL_EPOCHS,
    
    # Batch and image size
    batch=BATCH_SIZE,
    imgsz=IMAGE_SIZE,
    
    # Device
    device=0,  # Use GPU 0
    
    # Optimizer settings - fine-tuning with very low LR
    optimizer='AdamW',
    lr0=0.0001,  # Very low learning rate for fine-tuning
    lrf=1e-6,    # Final learning rate
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.03,
    
    # Augmentation - continue with strong augmentation
    hsv_h=0.02,      # HSV hue augmentation
    hsv_s=0.75,      # HSV saturation augmentation
    hsv_v=0.5,       # HSV value augmentation
    degrees=12.0,    # Rotation
    translate=0.2,   # Translation
    scale=0.8,       # Scaling
    shear=3.5,       # Shear
    perspective=0.001,  # Perspective
    flipud=0.0,      # Vertical flip
    fliplr=0.5,      # Horizontal flip
    bgr=0.0,         # BGR conversion
    mosaic=1.0,      # Mosaic augmentation
    mixup=0.25,      # Mixup augmentation
    copy_paste=0.15, # Copy-paste augmentation
    auto_augment='randaugment',  # Auto augmentation
    erasing=0.4,     # Random erasing
    crop_fraction=1.0,  # Crop fraction
    
    # Learning rate schedule
    cos_lr=True,     # Cosine learning rate schedule
    
    # Early stopping
    patience=50,     # Early stopping patience
    
    # Multi-scale training
    multi_scale=True,  # Multi-scale training
    
    # Validation
    val=True,
    
    # Save settings
    save=True,
    save_period=5,   # Save checkpoint every 5 epochs
    
    # Misc
    plots=True,
    verbose=True,
    exist_ok=True,
    project='runs/detect',
    name='continue_training_30more',
    workers=2,
    amp=True,  # Automatic Mixed Precision
)

# ===========================
# Training Complete
# ===========================
print()
print("=" * 90)
print("✅ TRAINING COMPLETED!")
print("=" * 90)
print()

# Parse results from CSV
try:
    results_csv = Path('runs/detect/continue_training_30more/results.csv')
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        # Get last epoch metrics
        last_epoch = df.iloc[-1]
        
        # Find best mAP50
        best_idx = df['metrics/mAP50(B)'].idxmax()
        best_epoch = df.iloc[best_idx]
        
        print("📊 Final Training Metrics:")
        print(f"   Final mAP@0.5:      {last_epoch['metrics/mAP50(B)']:.4f} ({last_epoch['metrics/mAP50(B)']*100:.2f}%)")
        print(f"   Best mAP@0.5:       {best_epoch['metrics/mAP50(B)']:.4f} ({best_epoch['metrics/mAP50(B)']*100:.2f}%) at epoch {int(best_epoch['                  epoch']) + 1}")
        print(f"   Final mAP@0.5:0.95: {last_epoch['metrics/mAP50-95(B)']:.4f} ({last_epoch['metrics/mAP50-95(B)']*100:.2f}%)")
        print(f"   Final Precision:    {last_epoch['metrics/precision(B)']:.4f} ({last_epoch['metrics/precision(B)']*100:.2f}%)")
        print(f"   Final Recall:       {last_epoch['metrics/recall(B)']:.4f} ({last_epoch['metrics/recall(B)']*100:.2f}%)")
        print()
except Exception as e:
    print(f"⚠️  Could not parse results CSV: {e}")
    print()

print(f"💾 New best model saved at: runs/detect/continue_training_30more/weights/best.pt")
print(f"📊 All results saved in: runs/detect/continue_training_30more/")
print()

# ===========================
# Validation
# ===========================
print("=" * 90)
print("🔍 VALIDATING NEW BEST MODEL")
print("=" * 90)

# Load and validate the new best model
best_model_path = 'runs/detect/continue_training_30more/weights/best.pt'
if Path(best_model_path).exists():
    best_model = YOLO(best_model_path)
    metrics = best_model.val(data=DATA_CONFIG, imgsz=IMAGE_SIZE, device=0)
    
    print()
    print("📋 PER-CLASS PERFORMANCE:")
    print("-" * 90)
    print(f"   {'Class':<25} {'AP@0.5':<20} Status")
    print("-" * 90)
    
    # Get per-class AP@0.5
    class_names = list(model.names.values())
    
    # metrics.box.ap50 contains per-class AP@0.5
    if hasattr(metrics.box, 'ap50'):
        for i, class_name in enumerate(class_names):
            ap50 = metrics.box.ap50[i]
            
            # Status based on performance
            if ap50 >= 0.85:
                status = "✅ Excellent"
            elif ap50 >= 0.75:
                status = "📈 Very Good"
            elif ap50 >= 0.65:
                status = "👍 Good"
            else:
                status = "⚠️ Needs work"
            
            print(f"   {class_name:<25} {ap50:.4f} ({ap50*100:.2f}%)  {status}")
    
    print()
else:
    print("⚠️  Best model not found for validation")

print()
print("=" * 90)
print("✅ CONTINUED TRAINING (30 MORE EPOCHS) COMPLETE!")
print("=" * 90)
print()
print(f"💡 Model ready to use: runs/detect/continue_training_30more/weights/best.pt")
print(f"💡 Total epochs trained: 50 (initial + 20 + 30)")
print("=" * 90)
