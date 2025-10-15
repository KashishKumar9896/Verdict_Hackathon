"""
Quick Image Detection Script
=============================
Usage: python detect_image.py <image_path>
Example: python detect_image.py test.jpg
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import sys

# Check if image path is provided
if len(sys.argv) < 2:
    print("❌ Error: No image path provided")
    print()
    print("Usage: python detect_image.py <image_path>")
    print()
    print("Examples:")
    print('  python detect_image.py "test.jpg"')
    print('  python detect_image.py "d:\\hackathon\\dataset\\hackathon2_test3\\test3\\images\\000000001_dark_clutter.png"')
    print()
    exit(1)

# Get image path from command line
IMAGE_PATH = sys.argv[1]

# Model and config
MODEL_PATH = r"runs/detect/continue_training_20epochs/weights/best.pt"
CONFIDENCE = 0.25

# ===========================
# Load Model
# ===========================
print("=" * 90)
print("🚀 YOLO OBJECT DETECTION")
print("=" * 90)
print()

if not Path(MODEL_PATH).exists():
    print(f"❌ Error: Model not found at {MODEL_PATH}")
    exit(1)

print(f"📦 Loading model...")
model = YOLO(MODEL_PATH)
print(f"✅ Model loaded!")
print()

# ===========================
# Check Image
# ===========================
if not Path(IMAGE_PATH).exists():
    print(f"❌ Error: Image not found at {IMAGE_PATH}")
    exit(1)

print(f"🖼️  Image: {Path(IMAGE_PATH).name}")
print()

# ===========================
# Run Detection
# ===========================
print(f"🔍 Detecting objects (confidence ≥ {CONFIDENCE*100:.0f}%)...")
print()

results = model.predict(
    source=IMAGE_PATH,
    conf=CONFIDENCE,
    iou=0.45,
    show=False,
    save=True,
    project='runs/detect',
    name='quick_detect',
    exist_ok=True
)

result = results[0]

# ===========================
# Display Results
# ===========================
print("📊 RESULTS:")
print("-" * 90)

if len(result.boxes) == 0:
    print("⚠️  No objects detected!")
else:
    print(f"✅ Found {len(result.boxes)} object(s):\n")
    
    for i, box in enumerate(result.boxes, 1):
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        
        print(f"  {i}. {class_name:<25} {confidence*100:>6.2f}%")

print()
print("-" * 90)
print()

# ===========================
# Show Image
# ===========================
print("🖼️  Displaying result... (Press any key to close)")
print()

annotated_img = result.plot()
cv2.imshow(f'Detection - {Path(IMAGE_PATH).name}', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Done! Result saved to: runs/detect/quick_detect/")
print("=" * 90)
