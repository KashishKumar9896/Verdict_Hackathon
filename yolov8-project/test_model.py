"""
Single Photo Detection Script
==============================
Test your trained YOLO model on a single image and see the detection results.
"""

from ultralytics import YOLO
import cv2
from pathlib import Path

# ===========================
# Configuration
# ===========================

# Use the latest trained model (20 epochs)
MODEL_PATH = r"runs/detect/continue_training_20epochs/weights/best.pt"

# Test image path - UPDATE THIS to test different images
IMAGE_PATH = r"d:\hackathon\dataset\hackathon2_test3\test3\images\000000021_dark_clutter.png"

# Detection confidence threshold
CONFIDENCE = 0.25

# ===========================
# Load Model
# ===========================
print("=" * 90)
print("🚀 SINGLE PHOTO DETECTION TEST")
print("=" * 90)
print()
print(f"📦 Loading model: {MODEL_PATH}")

if not Path(MODEL_PATH).exists():
    print(f"❌ Error: Model not found at {MODEL_PATH}")
    print("\nAvailable models:")
    print("  - runs/detect/continue_training_20epochs/weights/best.pt")
    print("  - runs/detect/final_long_training/weights/best.pt")
    exit(1)

model = YOLO(MODEL_PATH)
print(f"✅ Model loaded successfully!")
print(f"   Classes: {list(model.names.values())}")
print()

# ===========================
# Check Image
# ===========================
print(f"🖼️  Test image: {IMAGE_PATH}")

if not Path(IMAGE_PATH).exists():
    print(f"❌ Error: Image not found at {IMAGE_PATH}")
    print("\nTip: Update IMAGE_PATH variable in the script to point to your test image")
    print("Example test images available in:")
    print("  d:\\hackathon\\dataset\\hackathon2_test3\\test3\\images\\")
    exit(1)

print(f"✅ Image found!")
print()

# ===========================
# Run Detection
# ===========================
print("=" * 90)
print(f"🔍 RUNNING DETECTION (Confidence threshold: {CONFIDENCE})")
print("=" * 90)
print()

# Run inference
results = model.predict(
    source=IMAGE_PATH,
    conf=CONFIDENCE,
    iou=0.45,
    show=False,  # Don't show automatically, we'll display with annotations
    save=True,   # Save result
    project='runs/detect',
    name='test_single_photo',
    exist_ok=True
)

# ===========================
# Display Results
# ===========================
result = results[0]  # Get first (and only) result

print("📊 DETECTION RESULTS:")
print("-" * 90)

if len(result.boxes) == 0:
    print("⚠️  No objects detected in the image!")
    print(f"   Try lowering the confidence threshold (current: {CONFIDENCE})")
else:
    print(f"✅ Found {len(result.boxes)} object(s)!")
    print()
    print(f"{'#':<5} {'Class':<25} {'Confidence':<15} {'Bounding Box (x1, y1, x2, y2)'}")
    print("-" * 90)
    
    for i, box in enumerate(result.boxes, 1):
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        
        print(f"{i:<5} {class_name:<25} {confidence*100:>6.2f}%        "
              f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")

print()
print("-" * 90)

# ===========================
# Show Image with Detections
# ===========================
print()
print("🖼️  Displaying image with detections...")
print("   Press any key to close the window")
print()

# Get annotated image
annotated_img = result.plot()

# Display the image
cv2.imshow(f'Detection Results - {Path(IMAGE_PATH).name}', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===========================
# Summary
# ===========================
print("=" * 90)
print("✅ DETECTION COMPLETE!")
print("=" * 90)
print()
print(f"📁 Annotated image saved to: runs/detect/test_single_photo/")
print()
print("💡 To test another image:")
print("   1. Update IMAGE_PATH variable in this script")
print("   2. Run: python test_model.py")
print()
print("=" * 90)
