"""
YOLOv8 Fine-Tuning Script
Continue training from existing weights to improve accuracy
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    print("=" * 80)
    print("🎯 YOLO MODEL FINE-TUNING - CONTINUE TRAINING")
    print("=" * 80)
    
    # ========================================
    # STEP 1: LOAD PREVIOUS RESULTS
    # ========================================
    print("\n📊 Loading Previous Training Results...")
    
    # Previous model metrics (from your report)
    previous_metrics = {
        'mAP@0.5': 0.8296,
        'mAP@0.5:0.95': 0.7321,
        'Precision': 0.9363,
        'Recall': 0.693,
        'Epochs': 48
    }
    
    print(f"\n📈 Previous Performance (48 epochs):")
    print(f"   mAP@0.5:      {previous_metrics['mAP@0.5']:.4f} ({previous_metrics['mAP@0.5']*100:.2f}%)")
    print(f"   mAP@0.5:0.95: {previous_metrics['mAP@0.5:0.95']:.4f} ({previous_metrics['mAP@0.5:0.95']*100:.2f}%)")
    print(f"   Precision:    {previous_metrics['Precision']:.4f} ({previous_metrics['Precision']*100:.2f}%)")
    print(f"   Recall:       {previous_metrics['Recall']:.4f} ({previous_metrics['Recall']*100:.2f}%)")
    
    print(f"\n🎯 Target Goals:")
    print(f"   mAP@0.5:      > 0.90 (90%)")
    print(f"   Recall:       > 0.80 (80%)")
    
    # ========================================
    # STEP 2: CONFIGURATION
    # ========================================
    print("\n" + "=" * 80)
    print("📋 FINE-TUNING CONFIGURATION")
    print("=" * 80)
    
    # Fine-tuning parameters
    EPOCHS = 30
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    
    # Paths
    config_path = 'D:/hackathon/dataset/hackathon2_train_3/train_3/config.yaml'
    weights_path = 'D:/hackathon/Verdict_Hackathon/yolov8-project/runs/detect/final_validation/weights/best.pt'
    
    # Check if weights exist
    if not Path(weights_path).exists():
        print(f"\n⚠️ Weights not found at: {weights_path}")
        print("Trying alternative path...")
        weights_path = 'runs/detect/full_training/weights/best.pt'
        if Path(weights_path).exists():
            print(f"✅ Found weights at: {weights_path}")
        else:
            print("❌ No trained weights found. Please train a model first.")
            exit(1)
    
    print(f"\n✅ Configuration:")
    print(f"   Previous Weights: {weights_path}")
    print(f"   Config File:      {config_path}")
    print(f"   Additional Epochs: {EPOCHS}")
    print(f"   Batch Size:       {BATCH_SIZE}")
    print(f"   Image Size:       {IMAGE_SIZE}")
    
    # Load dataset config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n📂 Dataset:")
    print(f"   Training:   {config['train']}")
    print(f"   Validation: {config['val']}")
    print(f"   Classes ({config['nc']}): {config['names']}")
    
    # ========================================
    # STEP 3: LOAD MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("🔧 LOADING EXISTING MODEL")
    print("=" * 80)
    
    model = YOLO(weights_path)
    print(f"✅ Model loaded from: {weights_path}")
    print(f"   Using: NVIDIA GeForce RTX 3050 Laptop GPU")
    
    # ========================================
    # STEP 4: FINE-TUNE MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("🔥 STARTING FINE-TUNING (30 MORE EPOCHS)")
    print("=" * 80)
    print(f"\nThis will take approximately 20-25 minutes")
    print("Model will continue learning from epoch 49...")
    print("\nPress Ctrl+C to stop early (best model will still be saved)")
    print("=" * 80)
    
    try:
        # Fine-tune the model with optimized parameters
        results = model.train(
            # Dataset
            data=config_path,
            
            # Training settings
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            
            # Hardware
            device=0,  # Use GPU
            workers=2,
            
            # Learning rate tuning for fine-tuning
            optimizer='AdamW',  # Better for fine-tuning
            lr0=0.001,  # Lower learning rate for fine-tuning
            lrf=0.0001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=2,
            warmup_momentum=0.8,
            warmup_bias_lr=0.05,
            
            # Strong augmentation for better generalization
            hsv_h=0.02,  # Hue augmentation
            hsv_s=0.8,   # Saturation augmentation
            hsv_v=0.5,   # Value/Brightness augmentation
            degrees=5.0,  # Rotation augmentation
            translate=0.15,
            scale=0.6,
            shear=2.0,
            perspective=0.001,
            flipud=0.0,
            fliplr=0.5,  # Horizontal flip
            mosaic=1.0,
            mixup=0.1,   # Mix images for better learning
            copy_paste=0.0,
            
            # Validation
            val=True,
            save=True,
            save_period=5,
            patience=15,  # Stop if no improvement for 15 epochs
            
            # Output
            project='D:/hackathon/Verdict_Hackathon/yolov8-project/runs/detect',
            name='final_tuning',
            exist_ok=True,
            pretrained=False,  # We're continuing training
            verbose=True,
            plots=True,
        )
        
        print("\n" + "=" * 80)
        print("✅ FINE-TUNING COMPLETED!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Best model weights have been saved")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        exit(1)
    
    # ========================================
    # STEP 5: LOAD BEST FINE-TUNED MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("📦 LOADING BEST FINE-TUNED MODEL")
    print("=" * 80)
    
    best_model_path = Path('D:/hackathon/Verdict_Hackathon/yolov8-project/runs/detect/final_tuning/weights/best.pt')
    
    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        print(f"✅ Best fine-tuned model loaded from: {best_model_path}")
    else:
        print("⚠️ Using last trained model")
        best_model = model
    
    # ========================================
    # STEP 6: VALIDATE FINE-TUNED MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("🔍 VALIDATING FINE-TUNED MODEL")
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
        project='D:/hackathon/Verdict_Hackathon/yolov8-project/runs/detect',
        name='final_tuning_validation'
    )
    
    # ========================================
    # STEP 7: EXTRACT AND DISPLAY RESULTS
    # ========================================
    print("\n" + "=" * 80)
    print("📊 FINE-TUNED MODEL RESULTS")
    print("=" * 80)
    
    # Extract new metrics
    new_metrics = val_results.results_dict
    
    new_map50 = new_metrics.get('metrics/mAP50(B)', 0)
    new_map5095 = new_metrics.get('metrics/mAP50-95(B)', 0)
    new_precision = new_metrics.get('metrics/precision(B)', 0)
    new_recall = new_metrics.get('metrics/recall(B)', 0)
    new_f1 = 2 * new_precision * new_recall / (new_precision + new_recall + 1e-6)
    
    print("\n🎯 NEW PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"   mAP@0.5:          {new_map50:.4f} ({new_map50*100:.2f}%)")
    print(f"   mAP@0.5:0.95:     {new_map5095:.4f} ({new_map5095*100:.2f}%)")
    print(f"   Precision:        {new_precision:.4f} ({new_precision*100:.2f}%)")
    print(f"   Recall:           {new_recall:.4f} ({new_recall*100:.2f}%)")
    print(f"   F1-Score:         {new_f1:.4f} ({new_f1*100:.2f}%)")
    
    # ========================================
    # STEP 8: PER-CLASS PERFORMANCE
    # ========================================
    print("\n📋 PER-CLASS PERFORMANCE (AP@0.5):")
    print("-" * 80)
    print(f"   {'Class':<25} {'AP@0.5':<12} {'Percentage':<12}")
    print("-" * 80)
    
    if hasattr(val_results.box, 'ap50'):
        class_results = []
        for idx, ap in enumerate(val_results.box.ap50):
            class_name = best_model.names[idx]
            class_results.append((class_name, ap))
            print(f"   {class_name:<25} {ap:.4f}      {ap*100:>6.2f}%")
    
    # ========================================
    # STEP 9: IMPROVEMENT COMPARISON
    # ========================================
    print("\n" + "=" * 80)
    print("📈 IMPROVEMENT COMPARISON")
    print("=" * 80)
    
    improvements = {
        'mAP@0.5': ((new_map50 - previous_metrics['mAP@0.5']) / previous_metrics['mAP@0.5']) * 100,
        'mAP@0.5:0.95': ((new_map5095 - previous_metrics['mAP@0.5:0.95']) / previous_metrics['mAP@0.5:0.95']) * 100,
        'Precision': ((new_precision - previous_metrics['Precision']) / previous_metrics['Precision']) * 100,
        'Recall': ((new_recall - previous_metrics['Recall']) / previous_metrics['Recall']) * 100,
    }
    
    print("\n📊 Metric Improvements:")
    print("-" * 80)
    for metric, improvement in improvements.items():
        arrow = "📈" if improvement > 0 else "📉"
        sign = "+" if improvement > 0 else ""
        print(f"   {metric:<20} {arrow} {sign}{improvement:>6.2f}%")
    
    # Goal achievement
    print("\n🎯 Goal Achievement:")
    print("-" * 80)
    map_goal = "✅ ACHIEVED" if new_map50 >= 0.90 else f"❌ NOT YET (need {(0.90-new_map50)*100:.2f}% more)"
    recall_goal = "✅ ACHIEVED" if new_recall >= 0.80 else f"❌ NOT YET (need {(0.80-new_recall)*100:.2f}% more)"
    print(f"   mAP@0.5 > 90%:     {map_goal}")
    print(f"   Recall > 80%:      {recall_goal}")
    
    # ========================================
    # STEP 10: OPTIMIZATION RECOMMENDATIONS
    # ========================================
    print("\n" + "=" * 80)
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    if new_map50 < 0.90:
        diff = (0.90 - new_map50) * 100
        recommendations.append(f"1. Continue training for 20-30 more epochs (need {diff:.1f}% improvement)")
        recommendations.append("2. Try YOLOv8m (medium) model for better accuracy")
        recommendations.append("3. Increase training data augmentation")
    
    if new_recall < 0.80:
        diff = (0.80 - new_recall) * 100
        recommendations.append(f"4. Lower confidence threshold during inference (need {diff:.1f}% improvement)")
        recommendations.append("5. Add more training samples for underperforming classes")
        recommendations.append("6. Increase mosaic augmentation to 1.0")
    
    if new_precision > 0.95 and new_recall < 0.75:
        recommendations.append("7. Model is too conservative - adjust confidence threshold")
    
    if not recommendations:
        recommendations.append("✅ Model performance is excellent! Goals achieved.")
        recommendations.append("💡 Consider testing on real-world scenarios")
        recommendations.append("💡 Export model for deployment: model.export(format='onnx')")
    
    print("\n📝 Next Steps:")
    for rec in recommendations:
        print(f"   {rec}")
    
    # ========================================
    # STEP 11: SUMMARY REPORT
    # ========================================
    print("\n" + "=" * 80)
    print("📄 TRAINING SUMMARY REPORT")
    print("=" * 80)
    
    print(f"\n📌 Training Information:")
    print(f"   Base Model:        48 epochs (previous)")
    print(f"   Fine-Tuning:       30 epochs (new)")
    print(f"   Total Epochs:      78 epochs")
    print(f"   Batch Size:        {BATCH_SIZE}")
    print(f"   Optimizer:         AdamW")
    print(f"   Learning Rate:     0.001 → 0.0001")
    
    print(f"\n💾 Output Files:")
    print(f"   Model Weights:     runs/detect/final_tuning/weights/best.pt")
    print(f"   Training Plots:    runs/detect/final_tuning/")
    print(f"   Validation:        runs/detect/final_tuning_validation/")
    print(f"   Confusion Matrix:  runs/detect/final_tuning_validation/confusion_matrix.png")
    print(f"   PR Curves:         runs/detect/final_tuning_validation/BoxPR_curve.png")
    
    print("\n" + "=" * 80)
    print("✅ FINE-TUNING COMPLETE!")
    print("=" * 80)
    
    # Overall grade
    if new_map50 >= 0.90 and new_recall >= 0.80:
        grade = "EXCELLENT ⭐⭐⭐⭐⭐ - All goals achieved!"
    elif new_map50 >= 0.85 and new_recall >= 0.75:
        grade = "VERY GOOD ⭐⭐⭐⭐ - Close to goals!"
    elif new_map50 >= 0.80 and new_recall >= 0.70:
        grade = "GOOD ⭐⭐⭐ - Solid performance!"
    else:
        grade = "FAIR ⭐⭐ - More training needed"
    
    print(f"\n🏆 Final Grade: {grade}")
    print(f"   mAP@0.5: {new_map50*100:.2f}%")
    print(f"   Recall:  {new_recall*100:.2f}%")
    
    print("\n📍 Quick Test Command:")
    print(f"   yolo predict model=runs/detect/final_tuning/weights/best.pt source=your_image.jpg")
