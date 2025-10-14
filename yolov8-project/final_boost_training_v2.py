"""
YOLOv8 Final Boost Training Script v2
Advanced fine-tuning with optimized hyperparameters for maximum accuracy
Special focus on EmergencyPhone class and pushing mAP above 86%
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 YOLO FINAL BOOST TRAINING V2 - 50 EPOCHS")
    print("=" * 80)
    
    # Previous metrics baseline
    print("\n📊 Previous Performance:")
    print("   mAP@0.5:      83.76%")
    print("   Recall:       69.65%")
    print("   Precision:    94.19%")
    print("\n🎯 New Target: mAP@0.5 > 86%")
    print("🔍 Special Focus: EmergencyPhone class improvement")
    
    # Configuration
    EPOCHS = 30  # Total epochs
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    
    # Paths
    config_path = 'config/config.yaml'
    weights_path = 'runs/detect/full_training/weights/best.pt'
    
    # Check for better weights if available
    alternative_weights = [
        'runs/detect/final_tuning/weights/best.pt',
        'runs/detect/validation_results9/weights/best.pt',
    ]
    
    for alt_path in alternative_weights:
        if Path(alt_path).exists():
            weights_path = alt_path
            break
    
    print(f"\n✅ Configuration:")
    print(f"   Model Weights:    {weights_path}")
    print(f"   Config:           {config_path}")
    print(f"   Epochs:           {EPOCHS} total (resuming from epoch 20)")
    print(f"   Batch Size:       {BATCH_SIZE}")
    print(f"   Image Size:       {IMAGE_SIZE}")
    print(f"   Special Focus:    EmergencyPhone class")
    
    # Load model - resume from checkpoint if available
    checkpoint_path = 'runs/detect/final_boost_v2/weights/last.pt'
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Resuming from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        print(f"\n⚡ Starting fresh from: {weights_path}")
        model = YOLO(weights_path)
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Classes: {list(model.names.values())}")
    
    # Start training
    print("\n" + "=" * 80)
    print("🔥 STARTING ADVANCED FINE-TUNING - 30 EPOCHS (FROM EPOCH 20)")
    print("=" * 80)
    print("\nEnhancements for this run:")
    print("  ✓ Stronger augmentation (rotation, brightness, scale)")
    print("  ✓ Mixup augmentation (0.15) for better generalization")
    print("  ✓ Copy-paste augmentation (0.05) for hard examples")
    print("  ✓ Resuming from previous training (epoch 20)")
    print("  ✓ Lower learning rate (0.0008) for fine details")
    print("  ✓ Patience = 20 epochs")
    print("\nEstimated time: 20-25 minutes")
    print("=" * 80)
    
    results = model.train(
        # Dataset
        data=config_path,
        
        # Training settings
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        
        # Hardware
        device=0,
        workers=2,
        
        # Optimizer - AdamW is better for fine-tuning
        optimizer='AdamW',
        lr0=0.0008,  # Lower learning rate for fine-tuning
        lrf=0.00008,  # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,  # Longer warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,
        
        # STRONGER augmentation for better generalization
        hsv_h=0.025,  # Increased hue variation
        hsv_s=0.85,   # Increased saturation
        hsv_v=0.55,   # Increased brightness variation
        degrees=8.0,  # Increased rotation (was 5.0)
        translate=0.18,  # Increased translation (was 0.15)
        scale=0.7,    # Increased scale variation (was 0.6)
        shear=3.0,    # Increased shear (was 2.0)
        perspective=0.002,  # Slight perspective change
        flipud=0.0,   # No vertical flip for safety equipment
        fliplr=0.5,   # Horizontal flip
        
        # Advanced augmentation techniques
        mosaic=1.0,   # Always use mosaic
        mixup=0.15,   # Increased mixup (was 0.1) - blends images
        copy_paste=0.05,  # Copy-paste augmentation for hard examples
        
        # Validation and saving
        val=True,
        save=True,
        save_period=5,  # Save every 5 epochs
        patience=20,  # Increased patience (was 15)
        
        # Output
        project='runs/detect',
        name='final_boost_v2',
        exist_ok=True,
        pretrained=False,  # Continue from checkpoint
        verbose=True,
        plots=True,
    )
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED!")
    print("=" * 80)
    
    # Get final metrics
    print("\n📊 Final Training Metrics:")
    try:
        results_csv = Path('runs/detect/final_boost_v2/results.csv')
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
            
            if best_map50 >= 0.86:
                print(f"\n🎉 TARGET ACHIEVED! mAP@0.5 = {best_map50*100:.2f}% (> 86%)")
            else:
                improvement_needed = (0.86 - best_map50) * 100
                print(f"\n📈 Close to target! Need {improvement_needed:.2f}% more to reach 86%")
    except Exception as e:
        print(f"   Could not read final metrics: {e}")
    
    print(f"\n💾 Best model saved at: runs/detect/final_boost_v2/weights/best.pt")
    print(f"📊 All results saved in: runs/detect/final_boost_v2/")
    
    # Validate the model
    print("\n" + "=" * 80)
    print("🔍 VALIDATING BEST MODEL")
    print("=" * 80)
    
    best_model = YOLO('runs/detect/final_boost_v2/weights/best.pt')
    val_results = best_model.val(
        data=config_path,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=0.25,
        iou=0.5,
        device=0,
        plots=True,
    )
    
    # Per-class results
    print("\n📋 PER-CLASS PERFORMANCE:")
    print("-" * 70)
    print(f"   {'Class':<25} {'AP@0.5':<12} {'Status':<20}")
    print("-" * 70)
    
    if hasattr(val_results.box, 'ap50'):
        for idx, ap in enumerate(val_results.box.ap50):
            class_name = best_model.names[idx]
            status = "✅ Excellent" if ap > 0.85 else "📈 Improved" if ap > 0.80 else "⚠️ Needs work"
            
            # Highlight EmergencyPhone
            if class_name == 'EmergencyPhone':
                print(f"   {class_name:<25} {ap:.4f} ({ap*100:.2f}%)  {status} 🎯 FOCUS CLASS")
            else:
                print(f"   {class_name:<25} {ap:.4f} ({ap*100:.2f}%)  {status}")
    
    print("\n" + "=" * 80)
    print("✅ FINAL BOOST TRAINING V2 COMPLETE!")
    print("=" * 80)
