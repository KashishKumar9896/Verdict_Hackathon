"""
Complete YOLO Training and Evaluation Pipeline
Train a new model from scratch and evaluate its performance
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 YOLO MODEL - COMPLETE TRAINING & EVALUATION")
    print("=" * 80)
    
    # ========================================
    # STEP 1: CONFIGURATION
    # ========================================
    print("\n📋 Configuration:")
    
    # Training parameters
    EPOCHS = 30  # Reduced for faster training
    BATCH_SIZE = 16
    IMAGE_SIZE = 640
    MODEL_NAME = 'yolov8s.pt'  # Small model - good balance
    
    print(f"   Model: {MODEL_NAME}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Image Size: {IMAGE_SIZE}")
    
    # Load dataset config
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n📂 Dataset:")
    print(f"   Training: {config['train']}")
    print(f"   Validation: {config['val']}")
    print(f"   Classes ({config['nc']}): {config['names']}")
    
    # ========================================
    # STEP 2: INITIALIZE MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("🔧 INITIALIZING MODEL")
    print("=" * 80)
    
    model = YOLO(MODEL_NAME)
    print(f"✅ Model loaded: {MODEL_NAME}")
    print(f"   Using: NVIDIA GeForce RTX 3050 Laptop GPU")
    
    # ========================================
    # STEP 3: TRAIN MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("🔥 STARTING TRAINING")
    print("=" * 80)
    print(f"\nTraining will take approximately {EPOCHS * 0.8:.0f}-{EPOCHS * 1.2:.0f} minutes")
    print("You can monitor progress in real-time below...")
    print("\nPress Ctrl+C to stop early (best model will still be saved)")
    print("=" * 80)
    
    try:
        # Train the model
        results = model.train(
            # Dataset
            data=config_path,
            
            # Training settings
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            
            # Hardware
            device=0,  # Use GPU
            workers=2,  # Reduced to avoid memory issues
            
            # Optimization
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            
            # Validation
            val=True,
            save=True,
            save_period=10,
            patience=50,
            
            # Output
            project='runs/detect',
            name='full_training',
            exist_ok=True,
            pretrained=True,
            verbose=True,
            plots=True,
        )
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Best model weights have been saved")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        exit(1)
    
    # ========================================
    # STEP 4: LOAD BEST MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("📦 LOADING BEST MODEL FOR EVALUATION")
    print("=" * 80)
    
    best_model_path = Path('runs/detect/full_training/weights/best.pt')
    
    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        print(f"✅ Best model loaded from: {best_model_path}")
    else:
        print("⚠️ Using last trained model")
        best_model = model
    
    # ========================================
    # STEP 5: VALIDATE ON TEST SET
    # ========================================
    print("\n" + "=" * 80)
    print("🔍 RUNNING VALIDATION ON VALIDATION SET")
    print("=" * 80)
    
    val_results = best_model.val(
        data=config_path,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=0.25,
        iou=0.5,
        device=0,
        plots=True,
        save_json=True,
        project='runs/detect',
        name='final_validation'
    )
    
    # ========================================
    # STEP 6: DISPLAY COMPREHENSIVE RESULTS
    # ========================================
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS - COMPREHENSIVE REPORT")
    print("=" * 80)
    
    # Extract metrics
    metrics = val_results.results_dict
    
    print("\n🎯 OVERALL PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"   mAP@0.5:          {metrics.get('metrics/mAP50(B)', 0):.4f} ({metrics.get('metrics/mAP50(B)', 0)*100:.2f}%)")
    print(f"   mAP@0.5:0.95:     {metrics.get('metrics/mAP50-95(B)', 0):.4f} ({metrics.get('metrics/mAP50-95(B)', 0)*100:.2f}%)")
    print(f"   Precision:        {metrics.get('metrics/precision(B)', 0):.4f} ({metrics.get('metrics/precision(B)', 0)*100:.2f}%)")
    print(f"   Recall:           {metrics.get('metrics/recall(B)', 0):.4f} ({metrics.get('metrics/recall(B)', 0)*100:.2f}%)")
    print(f"   F1-Score:         {2 * metrics.get('metrics/precision(B)', 0) * metrics.get('metrics/recall(B)', 0) / (metrics.get('metrics/precision(B)', 0) + metrics.get('metrics/recall(B)', 0) + 1e-6):.4f}")
    
    print("\n📋 PER-CLASS PERFORMANCE:")
    print("-" * 80)
    print(f"   {'Class':<25} {'AP@0.5':<12} {'Images':<10}")
    print("-" * 80)
    
    # Get per-class results
    if hasattr(val_results.box, 'ap50'):
        for idx, ap in enumerate(val_results.box.ap50):
            class_name = best_model.names[idx]
            print(f"   {class_name:<25} {ap:.4f} ({ap*100:.2f}%)")
    
    print("\n📈 TRAINING STATISTICS:")
    print("-" * 80)
    
    # Read results from CSV if available
    results_csv = Path('runs/detect/full_training/results.csv')
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        print(f"   Total Epochs:     {len(df)}")
        print(f"   Best Epoch:       {df['metrics/mAP50(B)'].idxmax() + 1}")
        print(f"   Final Loss:       {df['train/box_loss'].iloc[-1]:.4f}")
        print(f"   Best mAP@0.5:     {df['metrics/mAP50(B)'].max():.4f} ({df['metrics/mAP50(B)'].max()*100:.2f}%)")
    
    print("\n💾 OUTPUT FILES:")
    print("-" * 80)
    print(f"   Model Weights:    runs/detect/full_training/weights/best.pt")
    print(f"   Training Plots:   runs/detect/full_training/")
    print(f"   Validation:       runs/detect/final_validation/")
    print(f"   Confusion Matrix: runs/detect/final_validation/confusion_matrix.png")
    print(f"   PR Curves:        runs/detect/final_validation/BoxPR_curve.png")
    print(f"   F1 Curves:        runs/detect/final_validation/BoxF1_curve.png")
    print(f"   Results CSV:      runs/detect/full_training/results.csv")
    
    # ========================================
    # STEP 7: PERFORMANCE SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("🏆 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    map50 = metrics.get('metrics/mAP50(B)', 0) * 100
    
    if map50 >= 90:
        grade = "EXCELLENT ⭐⭐⭐⭐⭐"
    elif map50 >= 80:
        grade = "VERY GOOD ⭐⭐⭐⭐"
    elif map50 >= 70:
        grade = "GOOD ⭐⭐⭐"
    elif map50 >= 60:
        grade = "FAIR ⭐⭐"
    else:
        grade = "NEEDS IMPROVEMENT ⭐"
    
    print(f"\n   Your Model Grade: {grade}")
    print(f"   Accuracy: {map50:.2f}%")
    
    print("\n💡 NEXT STEPS:")
    print("-" * 80)
    print("   1. Review confusion matrix to see which classes need improvement")
    print("   2. Check PR curves to understand precision-recall tradeoff")
    print("   3. Test on new images: model.predict('image.jpg')")
    print("   4. Export model: model.export(format='onnx')")
    
    print("\n" + "=" * 80)
    print("✅ ALL DONE! YOUR MODEL IS READY TO USE!")
    print("=" * 80)
    
    print(f"\n📍 Quick Test Command:")
    print(f"   yolo predict model=runs/detect/full_training/weights/best.pt source=your_image.jpg")
