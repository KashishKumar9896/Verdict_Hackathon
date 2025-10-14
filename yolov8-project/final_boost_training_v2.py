"""
YOLOv8 Final Boost Training Script v2
Advanced fine-tuning with optimized hyperparameters for maximum accuracy
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 YOLO FINAL BOOST TRAINING V2")
    print("=" * 80)
    
    # Configuration
    EPOCHS = 30
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    
    # Paths
    config_path = 'config/config.yaml'
    weights_path = 'runs/detect/full_training/weights/best.pt'
    
    print(f"\n✅ Configuration:")
    print(f"   Model Weights:    {weights_path}")
    print(f"   Config:           {config_path}")
    print(f"   Epochs:           {EPOCHS}")
    print(f"   Batch Size:       {BATCH_SIZE}")
    print(f"   Image Size:       {IMAGE_SIZE}")
    
    # Load model
    model = YOLO(weights_path)
    print(f"\n✅ Model loaded successfully!")
    
    # Start training
    print("\n🔥 Starting fine-tuning...")
    
    results = model.train(
        data=config_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=0,
        workers=2,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.15,
        scale=0.6,
        shear=2.0,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        val=True,
        save=True,
        save_period=5,
        patience=15,
        project='runs/detect',
        name='final_boost_v2',
        exist_ok=True,
        pretrained=False,
        verbose=True,
        plots=True,
    )
    
    print("\n✅ Training completed!")
    print(f"   Best model saved at: runs/detect/final_boost_v2/weights/best.pt")
