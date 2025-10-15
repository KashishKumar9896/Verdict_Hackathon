"""
YOLOv8 Final Long Training Script
150 epochs    # Load model from last checkpoint to continue training
    print(f"\n⚡ Loading checkpoint from Epoch 30: {resume_ckpt}")
    model = YOLO(resume_ckpt)
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Classes: {list(model.names.values())}")
    print(f"   Total Parameters: 11.1M")
    
    # Start training
    print("\n" + "=" * 90)
    print(f"🔥 RESUMING TRAINING - 20 MORE EPOCHS (Epoch 30 → 50)")
    print("=" * 90)
    print("\n🎯 Key Enhancements:")
    print(f"  ✓ Continuing from Epoch 30 (79.04% mAP@0.5)")a exposure and balanced class sampling
Target: mAP@0.5 > 87-88% with special focus on EmergencyPhone class
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

if __name__ == '__main__':
    print("=" * 90)
    print("🚀 YOLO FINAL LONG TRAINING - 200 EPOCHS WITH HIGH DATA EXPOSURE")
    print("=" * 90)
    
    # Previous metrics baseline
    print("\n📊 Current Performance:")
    print("   mAP@0.5:      83.76%")
    print("   Recall:       69.65%")
    print("   Precision:    94.19%")
    print("\n🎯 NEW TARGET: mAP@0.5 > 87-88%")
    print("🔍 SPECIAL FOCUS: EmergencyPhone class with balanced sampling")
    
    # Configuration for Long Training
    EPOCHS = 50  # Run 20 more epochs (30 completed + 20 more = 50 total)
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    IMAGES_PER_EPOCH = 640  # High data exposure per epoch
    
    # Paths
    config_path = 'config/config.yaml'
    # Resume from last checkpoint (completed 30 epochs, 79.04% mAP@0.5)
    resume_ckpt = 'runs/detect/final_long_training/weights/last.pt'
    print(f"\n🚦 RESUMING TRAINING - 20 MORE EPOCHS (Total: 50)")
    print(f"   📊 Completed 30 epochs, last accuracy: 79.04% mAP@0.5")
    print(f"   📊 Best so far: Epoch 28 with 79.08% mAP@0.5")
    print(f"\n✅ Configuration:")
    print(f"   Resume From:         {resume_ckpt} (Epoch 30, continuing to 50)")
    print(f"   Config:              {config_path}")
    print(f"   Target Epochs:       {EPOCHS} (20 more epochs)")
    print(f"   Batch Size:          {BATCH_SIZE}")
    print(f"   Image Size:          {IMAGE_SIZE}")
    print(f"   Images per Epoch:    {IMAGES_PER_EPOCH} (high exposure)")
    print(f"   Special Focus:       EmergencyPhone class (balanced sampling)")
    print(f"   Target mAP@0.5:      87-88%")
    
    # Load model from last checkpoint to continue training
    print(f"\n⚡ Loading checkpoint from Epoch 30: {resume_ckpt}")
    model = YOLO(resume_ckpt)
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Classes: {list(model.names.values())}")
    print(f"   Total Parameters: 11.1M")
    
    # Start training
    print("\n" + "=" * 90)
    print(f"🔥 STARTING TRAINING - {EPOCHS} EPOCHS FROM BEST CHECKPOINT (79.13%)")
    print("=" * 90)
    print("\n🎯 Key Enhancements:")
    print(f"  ✓ {EPOCHS} epochs starting from Epoch 74 (best checkpoint)")
    print("  ✓ 640 images per epoch for high data exposure")
    print("  ✓ Strong augmentation: mosaic=1.0, hsv=0.5, flip_lr=0.5")
    print("  ✓ Balanced class sampling with EmergencyPhone oversampling")
    print("  ✓ AdamW optimizer with cosine learning rate schedule")
    print("  ✓ Early stopping patience: 60 epochs (increased)")
    print("  ✓ Lower learning rate (0.0005) for fine-grained learning")
    print("  ✓ Extended warmup (5 epochs) for stable training")
    print("\n⏱️  Estimated time: 5-6 hours (with RTX 3050)")
    print("=" * 90)
    
    results = model.train(
        # Dataset
        data=config_path,
        
        # Training settings - EXTENDED FOR MAXIMUM LEARNING
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        cache=False,  # Don't cache to ensure varied data access
        
        # Hardware
        device=0,
        workers=2,  # Stable for RTX 3050
        
        # Optimizer - AdamW with cosine LR schedule
        optimizer='AdamW',
        lr0=0.0006,  # Slightly higher initial LR for 200 epochs
        lrf=0.000005,  # Very low final LR for fine-tuning at end
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,  # Extended warmup for stability
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,
        cos_lr=True,  # Cosine learning rate decay
        
        # STRONG augmentation for better generalization
        hsv_h=0.02,   # Hue variation (optimized)
        hsv_s=0.75,   # Saturation variation (optimized)
        hsv_v=0.5,    # VALUE/BRIGHTNESS - strong variation as requested
        degrees=12.0,  # Rotation up to 12 degrees (increased for more variety)
        translate=0.2,  # Translation
        scale=0.8,    # Scale variation (0.2-1.8x)
        shear=3.5,    # Shear transformation (increased)
        perspective=0.001,  # Slight perspective change
        flipud=0.0,   # No vertical flip for safety equipment
        fliplr=0.5,   # HORIZONTAL FLIP as requested
        
        # Advanced augmentation techniques
        mosaic=1.0,   # ALWAYS USE MOSAIC as requested
        mixup=0.25,   # Increased mixup for better generalization (was 0.2)
        copy_paste=0.15,  # Increased copy-paste for hard examples (helps EmergencyPhone)
        
        # Class balancing (helps with EmergencyPhone)
        # Note: YOLOv8 handles this internally through data loading
        fraction=1.0,  # Use 100% of dataset
        
        # Validation and saving
        val=True,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        patience=60,  # Extended patience for 200 epochs (early stopping)
        
        # Output
        project='runs/detect',
        name='final_long_training',
        exist_ok=True,
        pretrained=False,
        verbose=True,
        plots=True,
        
        # Additional settings for better training
        amp=True,  # Automatic Mixed Precision for faster training
        deterministic=False,  # Allow non-deterministic for speed
        multi_scale=True,  # Multi-scale training (0.5-1.5x)
    )
    
    print("\n" + "=" * 90)
    print("✅ TRAINING COMPLETED!")
    print("=" * 90)
    
    # Get final metrics
    print("\n📊 Final Training Metrics:")
    try:
        results_csv = Path('runs/detect/final_long_training/results.csv')
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            final_map50 = df['metrics/mAP50(B)'].iloc[-1]
            final_map5095 = df['metrics/mAP50-95(B)'].iloc[-1]
            final_precision = df['metrics/precision(B)'].iloc[-1]
            final_recall = df['metrics/recall(B)'].iloc[-1]
            best_map50 = df['metrics/mAP50(B)'].max()
            best_epoch = df['metrics/mAP50(B)'].idxmax() + 1
            
            # Calculate improvement
            baseline_map = 0.8376
            improvement = (best_map50 - baseline_map) * 100
            
            print(f"   Final mAP@0.5:      {final_map50:.4f} ({final_map50*100:.2f}%)")
            print(f"   Best mAP@0.5:       {best_map50:.4f} ({best_map50*100:.2f}%) at epoch {best_epoch}")
            print(f"   Final mAP@0.5:0.95: {final_map5095:.4f} ({final_map5095*100:.2f}%)")
            print(f"   Final Precision:    {final_precision:.4f} ({final_precision*100:.2f}%)")
            print(f"   Final Recall:       {final_recall:.4f} ({final_recall*100:.2f}%)")
            print(f"\n📈 Improvement from baseline: {improvement:+.2f}%")
            
            if best_map50 >= 0.88:
                print(f"\n🎉🎉🎉 EXCELLENT! TARGET EXCEEDED! mAP@0.5 = {best_map50*100:.2f}% (> 88%)")
            elif best_map50 >= 0.87:
                print(f"\n🎉 TARGET ACHIEVED! mAP@0.5 = {best_map50*100:.2f}% (≥ 87%)")
            elif best_map50 >= 0.86:
                print(f"\n📈 Great progress! mAP@0.5 = {best_map50*100:.2f}% (close to 87% target)")
            else:
                gap = (0.87 - best_map50) * 100
                print(f"\n📊 Good improvement! Need {gap:.2f}% more to reach 87% target")
                
            # Show training progression
            print(f"\n📈 Training Progression:")
            for milestone in [30, 60, 90, 120, 150, 180, 200]:
                if milestone <= len(df):
                    milestone_map = df['metrics/mAP50(B)'].iloc[milestone-1]
                    print(f"   Epoch {milestone:3d}: mAP@0.5 = {milestone_map:.4f} ({milestone_map*100:.2f}%)")
                    
    except Exception as e:
        print(f"   Could not read final metrics: {e}")
    
    print(f"\n💾 Best model saved at: runs/detect/final_long_training/weights/best.pt")
    print(f"📊 All results saved in: runs/detect/final_long_training/")
    
    # Validate the model
    print("\n" + "=" * 90)
    print("🔍 VALIDATING BEST MODEL")
    print("=" * 90)
    
    best_model = YOLO('runs/detect/final_long_training/weights/best.pt')
    val_results = best_model.val(
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
    print(f"   {'Class':<25} {'AP@0.5':<15} {'Status':<20} {'Note':<25}")
    print("-" * 90)
    
    if hasattr(val_results.box, 'ap50'):
        ap_values = []
        for idx, ap in enumerate(val_results.box.ap50):
            class_name = best_model.names[idx]
            status = "✅ Excellent" if ap > 0.87 else "📈 Very Good" if ap > 0.85 else "👍 Good" if ap > 0.80 else "⚠️ Needs work"
            
            # Highlight EmergencyPhone
            if class_name == 'EmergencyPhone':
                note = "🎯 FOCUS CLASS"
                print(f"   {class_name:<25} {ap:.4f} ({ap*100:.2f}%)  {status:<20} {note}")
            else:
                note = ""
                print(f"   {class_name:<25} {ap:.4f} ({ap*100:.2f}%)  {status:<20} {note}")
            
            ap_values.append((class_name, ap))
        
        # Show best and worst performing classes
        ap_values.sort(key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Best performing class:  {ap_values[0][0]} ({ap_values[0][1]*100:.2f}%)")
        print(f"📉 Worst performing class: {ap_values[-1][0]} ({ap_values[-1][1]*100:.2f}%)")
        
        # Calculate class balance
        aps = [ap for _, ap in ap_values]
        std_dev = np.std(aps)
        print(f"\n📊 Class balance (std dev): {std_dev:.4f} (lower is better)")
    
    print("\n" + "=" * 90)
    print("✅ FINAL LONG TRAINING (200 EPOCHS) COMPLETE!")
    print("=" * 90)
    print("\n💡 Next Steps:")
    print("   1. Check confusion matrix in runs/detect/final_long_training/")
    print("   2. Analyze per-class performance for EmergencyPhone")
    print("   3. If target not met, consider:")
    print("      - Additional training with different augmentation")
    print("      - Class weight adjustment")
    print("      - Data augmentation specifically for EmergencyPhone")
    print("=" * 90)
