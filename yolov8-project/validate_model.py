"""
YOLO Model Validation Script
Initialize and validate the trained model to check mAP accuracy
"""

from ultralytics import YOLO
import os
from pathlib import Path

if __name__ == '__main__':
    print("=" * 80)
    print("🎯 YOLO MODEL VALIDATION - mAP ACCURACY CHECK")
    print("=" * 80)

    # ========================================
    # STEP 1: LOCATE TRAINED MODEL
    # ========================================
    print("\n📂 Locating trained model...")

    # Path to the best trained model
    model_path = Path("d:/hackathon/dataset/hackathon2_train_3/train_3/runs/detect/train/weights/best.pt")

    if not model_path.exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("\nTrying alternative paths...")
        
        # Try alternative paths
        alt_paths = [
            "runs/detect/train/weights/best.pt",
            "runs/detect/train5/weights/best.pt",
            "runs/detect/train4/weights/best.pt",
        ]
        
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                model_path = Path(alt_path)
                print(f"✅ Found model at: {model_path}")
                break
        else:
            print("\n❌ No trained model found. Please train a model first.")
            exit(1)
    else:
        print(f"✅ Model found: {model_path}")

    # ========================================
    # STEP 2: LOAD MODEL
    # ========================================
    print("\n🔧 Loading model...")

    try:
        model = YOLO(str(model_path))
        print("✅ Model loaded successfully!")
        print(f"\n📊 Model Information:")
        print(f"   Classes: {list(model.names.values())}")
        print(f"   Number of classes: {len(model.names)}")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        exit(1)

    # ========================================
    # STEP 3: VALIDATE MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("🔍 RUNNING VALIDATION ON TEST DATASET")
    print("=" * 80)
    print("\nThis will calculate:")
    print("  - mAP@0.5 (mean Average Precision at IoU=0.5)")
    print("  - mAP@0.5:0.95 (mAP across multiple IoU thresholds)")
    print("  - Precision, Recall, F1-Score per class")
    print("\nPlease wait, this may take a few minutes...\n")

    # Config file path
    config_path = "config/config.yaml"

    try:
        # Run validation
        results = model.val(
            data=config_path,
            imgsz=640,
            batch=16,       # Batch size for GPU
            conf=0.25,      # Confidence threshold
            iou=0.5,        # IoU threshold for NMS
            device='0',     # Use GPU 0 (RTX 3050)
            save_json=True,
            plots=True,
            verbose=True,
            project='runs/detect',
            name='validation_results'
        )
        
        # ========================================
        # STEP 4: DISPLAY RESULTS
        # ========================================
        print("\n" + "=" * 80)
        print("📈 VALIDATION RESULTS")
        print("=" * 80)
        
        # Extract metrics
        metrics = results.results_dict
        
        print(f"\n🎯 Overall Performance:")
        print(f"   mAP@0.5:      {metrics.get('metrics/mAP50(B)', 0):.4f} ({metrics.get('metrics/mAP50(B)', 0)*100:.2f}%)")
        print(f"   mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f} ({metrics.get('metrics/mAP50-95(B)', 0)*100:.2f}%)")
        print(f"   Precision:    {metrics.get('metrics/precision(B)', 0):.4f} ({metrics.get('metrics/precision(B)', 0)*100:.2f}%)")
        print(f"   Recall:       {metrics.get('metrics/recall(B)', 0):.4f} ({metrics.get('metrics/recall(B)', 0)*100:.2f}%)")
        
        print(f"\n📊 Additional Metrics:")
        print(f"   Fitness:      {metrics.get('fitness', 0):.4f}")
        
        # Per-class results
        if hasattr(results, 'ap_class_index'):
            print(f"\n📋 Per-Class Results:")
            print(f"   {'Class':<25} {'AP@0.5':<10}")
            print(f"   {'-'*35}")
            
            # This would need the per-class AP values which are available in results.box
            if hasattr(results.box, 'ap50'):
                for idx, ap in enumerate(results.box.ap50):
                    class_name = model.names[idx]
                    print(f"   {class_name:<25} {ap:.4f}")
        
        print(f"\n💾 Results saved to: runs/detect/validation_results/")
        print(f"   - Confusion matrix")
        print(f"   - PR curves")
        print(f"   - F1 curves")
        print(f"   - Prediction visualizations")
        
        print("\n" + "=" * 80)
        print("✅ VALIDATION COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that the dataset paths in config.yaml are correct")
        print("2. Ensure validation images exist in the val directory")
        print("3. Try reducing batch size if you get memory errors")
        exit(1)

    print("\n💡 Next Steps:")
    print("   - Check visualization results in runs/detect/validation_results/")
    print("   - Compare with training metrics")
    print("   - Use model for predictions: model.predict('image.jpg')")
