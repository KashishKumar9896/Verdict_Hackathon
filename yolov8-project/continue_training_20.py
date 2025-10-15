"""
Continue Training - 20 More Epochs from Best Model
Continue from the best checkpoint to push mAP higher
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    print("=" * 90)
    print("🚀 CONTINUING TRAINING - 20 MORE EPOCHS FROM BEST MODEL")
    print("=" * 90)
    
    # Configuration
    EPOCHS = 20
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    
    # Paths
    config_path = 'config/config.yaml'
    
    # Find the best model
    best_model_paths = [
        'runs/detect/final_long_training/weights/best.pt',
        'd:/hackathon/dataset/hackathon2_train_3/train_3/runs/detect/train/weights/best.pt'
    ]
    
    best_model_path = None
    for path in best_model_paths:
        if os.path.exists(path):
            best_model_path = path
            print(f"✅ Found best model: {path}")
            break
    
    if not best_model_path:
        print("❌ Best model not found! Checked:")
        for path in best_model_paths:
            print(f"   - {path}")
        exit(1)
    
    print(f"\n✅ Configuration:")
    print(f"   Model:               {best_model_path}")
    print(f"   Config:              {config_path}")
    print(f"   Additional Epochs:   {EPOCHS}")
    print(f"   Batch Size:          {BATCH_SIZE}")
    print(f"   Image Size:          {IMAGE_SIZE}")
    
    # Load model
    print(f"\n⚡ Loading best model...")
    model = YOLO(best_model_path)
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Classes: {list(model.names.values())}")
    
    # Start training
    print("\n" + "=" * 90)
    print(f"🔥 RESUMING TRAINING - {EPOCHS} MORE EPOCHS")
    print("=" * 90)
    print("\n🎯 Training Settings:")
    print("  ✓ Strong augmentation for better generalization")
    print("  ✓ Mosaic augmentation enabled")
    print("  ✓ AdamW optimizer with cosine LR")
    print("  ✓ Lower learning rate for fine-tuning")
    print("  ✓ Early stopping patience: 50")
    print("\n⏱️  Estimated time: ~2 hours")
    print("=" * 90)
    
    results = model.train(
        # Dataset
        data=config_path,
        
        # Training settings
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        cache=False,
        
        # Hardware
        device=0,
        workers=2,
        
        # Optimizer - Fine-tuning settings
        optimizer='AdamW',
        lr0=0.0003,  # Lower LR for fine-tuning
        lrf=0.000001,  # Very low final LR
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.03,
        cos_lr=True,
        
        # Strong augmentation
        hsv_h=0.02,
        hsv_s=0.75,
        hsv_v=0.5,
        degrees=12.0,
        translate=0.2,
        scale=0.8,
        shear=3.5,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        
        # Advanced augmentation
        mosaic=1.0,
        mixup=0.25,
        copy_paste=0.15,
        
        # Data
        fraction=1.0,
        
        # Validation and saving
        val=True,
        save=True,
        save_period=5,
        patience=50,
        
        # Output
        project='runs/detect',
        name='continue_training_20epochs',
        exist_ok=True,
        pretrained=False,
        verbose=True,
        plots=True,
        
        # Additional settings
        amp=True,
        deterministic=False,
        multi_scale=True,
    )
    
    print("\n" + "=" * 90)
    print("✅ TRAINING COMPLETED!")
    print("=" * 90)
    
    # Get final metrics
    print("\n📊 Final Training Metrics:")
    try:
        results_csv = Path('runs/detect/continue_training_20epochs/results.csv')
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            final_map50 = df['metrics/mAP50(B)'].iloc[-1]
            final_map5095 = df['metrics/mAP50-95(B)'].iloc[-1]
            final_precision = df['metrics/precision(B)'].iloc[-1]
            final_recall = df['metrics/recall(B)'].iloc[-1]
            best_map50 = df['metrics/mAP50(B)'].max()
            best_epoch = df['metrics/mAP50(B)'].idxmax() + 1
            
            print(f"   Final mAP@0.5:      {final_map50:.4f} ({final_map50*100:.2f}%)")
            print(f"   Best mAP@0.5:       {best_map50:.4f} ({best_map50*100:.2f}%) at epoch {best_epoch}")
            print(f"   Final mAP@0.5:0.95: {final_map5095:.4f} ({final_map5095*100:.2f}%)")
            print(f"   Final Precision:    {final_precision:.4f} ({final_precision*100:.2f}%)")
            print(f"   Final Recall:       {final_recall:.4f} ({final_recall*100:.2f}%)")
            
    except Exception as e:
        print(f"   Could not read final metrics: {e}")
    
    print(f"\n💾 New best model saved at: runs/detect/continue_training_20epochs/weights/best.pt")
    print(f"📊 All results saved in: runs/detect/continue_training_20epochs/")
    
    # Validate the model
    print("\n" + "=" * 90)
    print("🔍 VALIDATING NEW BEST MODEL")
    print("=" * 90)
    
    new_best_model = YOLO('runs/detect/continue_training_20epochs/weights/best.pt')
    val_results = new_best_model.val(
        data=config_path,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=0.25,
        iou=0.5,
        device=0,
        plots=True,
        save_json=True,
    )
    
    # Per-class results
    print("\n📋 PER-CLASS PERFORMANCE:")
    print("-" * 90)
    print(f"   {'Class':<25} {'AP@0.5':<15} {'Status':<20}")
    print("-" * 90)
    
    if hasattr(val_results.box, 'ap50'):
        for idx, ap in enumerate(val_results.box.ap50):
            class_name = new_best_model.names[idx]
            status = "✅ Excellent" if ap > 0.87 else "📈 Very Good" if ap > 0.85 else "👍 Good" if ap > 0.80 else "⚠️ Needs work"
            print(f"   {class_name:<25} {ap:.4f} ({ap*100:.2f}%)  {status:<20}")
    
    print("\n" + "=" * 90)
    print("✅ CONTINUED TRAINING (20 EPOCHS) COMPLETE!")
    print("=" * 90)
    print(f"\n💡 Model ready to use: runs/detect/continue_training_20epochs/weights/best.pt")
    print("=" * 90)
