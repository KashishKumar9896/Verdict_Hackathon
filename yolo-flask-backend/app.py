"""
Flask Backend for YOLO Real-Time Detection
Receives camera frames from Expo app and returns detection results
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native app

# Request counter
request_counter = {'total': 0, 'success': 0, 'failed': 0}

# Load YOLO model at startup
logger.info("🚀 Loading YOLO model...")
MODEL_PATH = "../yolov8-project/runs/detect/final_long_training/weights/best.pt"
try:
    model = YOLO(MODEL_PATH)
    logger.info("✅ YOLO model loaded successfully!")
    logger.info(f"📊 Model classes: {model.names}")
    logger.info(f"📍 Model path: {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    raise

# Configuration
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("🏥 Health check requested")
    client_ip = request.remote_addr
    logger.info(f"   Client IP: {client_ip}")
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'classes': list(model.names.values()),
        'total_requests': request_counter['total'],
        'success_count': request_counter['success'],
        'failed_count': request_counter['failed'],
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/detect', methods=['POST'])
def detect():
    """
    Detect objects in uploaded image
    Expects: JSON with base64 encoded image
    Returns: Detection results with bounding boxes and labels
    """
    request_counter['total'] += 1
    request_id = request_counter['total']
    client_ip = request.remote_addr
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📸 NEW REQUEST #{request_id} from {client_ip}")
    logger.info(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Get image from request
        data = request.get_json()
        
        if 'image' not in data:
            logger.error(f"❌ Request #{request_id}: No image provided in request")
            request_counter['failed'] += 1
            return jsonify({'error': 'No image provided'}), 400
        
        logger.info(f"✅ Request #{request_id}: Image data received")
        
        # Decode base64 image
        image_data = data['image']
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Log image data size
        image_size_kb = len(image_data) / 1024
        logger.info(f"📦 Request #{request_id}: Image size: {image_size_kb:.2f} KB (base64)")
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        logger.info(f"🔓 Request #{request_id}: Base64 decoded successfully")
        
        image = Image.open(BytesIO(image_bytes))
        logger.info(f"🖼️  Request #{request_id}: Image opened - Size: {image.size}, Mode: {image.mode}")
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV (if needed)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        logger.info(f"🔄 Request #{request_id}: Image converted to numpy array - Shape: {img_array.shape}")
        logger.info(f"🤖 Request #{request_id}: Running YOLO inference...")
        
        # Run YOLO detection
        inference_start = time.time()
        results = model.predict(
            img_array,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        inference_time = time.time() - inference_start
        
        logger.info(f"⚡ Request #{request_id}: Inference completed in {inference_time:.3f}s")
        
        # Parse results
        detections = []
        for r in results:
            boxes = r.boxes
            logger.info(f"📊 Request #{request_id}: Found {len(boxes)} detections")
            
            for idx, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence and class
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                logger.info(f"   Detection {idx+1}: {class_name} ({confidence:.2%}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                
                detections.append({
                    'class': class_name,
                    'confidence': round(confidence, 2),
                    'bbox': {
                        'x1': round(x1),
                        'y1': round(y1),
                        'x2': round(x2),
                        'y2': round(y2),
                        'width': round(x2 - x1),
                        'height': round(y2 - y1)
                    }
                })
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Request #{request_id}: SUCCESS - {len(detections)} objects detected")
        logger.info(f"⏱️  Request #{request_id}: Total processing time: {processing_time:.3f}s")
        logger.info(f"{'='*60}\n")
        
        request_counter['success'] += 1
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'processing_time': round(processing_time, 3),
            'inference_time': round(inference_time, 3),
            'request_id': request_id,
            'image_size': {
                'width': img_array.shape[1],
                'height': img_array.shape[0]
            }
        })
    
    except Exception as e:
        logger.error(f"❌ Request #{request_id}: ERROR - {str(e)}")
        logger.error(f"{'='*60}\n")
        request_counter['failed'] += 1
        return jsonify({
            'success': False,
            'error': str(e),
            'request_id': request_id
        }), 500

@app.route('/detect-stream', methods=['POST'])
def detect_stream():
    """
    Optimized endpoint for real-time video streaming
    Lower quality processing for faster response
    """
    request_counter['total'] += 1
    request_id = request_counter['total']
    
    logger.info(f"🎥 STREAM REQUEST #{request_id} from {request.remote_addr}")
    
    try:
        start_time = time.time()
        
        data = request.get_json()
        if 'image' not in data:
            logger.error(f"❌ Stream #{request_id}: No image provided")
            request_counter['failed'] += 1
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Resize for faster processing (optional)
        max_size = 640
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        logger.info(f"🎬 Stream #{request_id}: Processing image {img_array.shape}")
        
        # Run detection with lower settings for speed
        results = model.predict(
            img_array,
            conf=0.3,  # Slightly higher confidence for speed
            iou=0.5,
            imgsz=416,  # Smaller image size for speed
            verbose=False,
            half=True  # Use FP16 if available
        )
        
        # Parse results (simplified)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                detections.append({
                    'class': class_name,
                    'confidence': round(confidence, 2),
                    'x': round((x1 + x2) / 2),  # Center X
                    'y': round((y1 + y2) / 2),  # Center Y
                    'w': round(x2 - x1),
                    'h': round(y2 - y1)
                })
        
        processing_time = time.time() - start_time
        fps = round(1 / processing_time, 1) if processing_time > 0 else 0
        
        logger.info(f"✅ Stream #{request_id}: {len(detections)} objects - {fps} FPS")
        request_counter['success'] += 1
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'fps': fps
        })
    
    except Exception as e:
        logger.error(f"❌ Stream #{request_id}: ERROR - {str(e)}")
        request_counter['failed'] += 1
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint for mobile app (simplified response)
    """
    request_counter['total'] += 1
    request_id = request_counter['total']
    client_ip = request.remote_addr
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📱 MOBILE PREDICT #{request_id} from {client_ip}")
    logger.info(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        data = request.get_json()
        if 'image' not in data:
            logger.error(f"❌ Predict #{request_id}: No image in request")
            request_counter['failed'] += 1
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        logger.info(f"✅ Predict #{request_id}: Image data received")
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_size_kb = len(image_data) / 1024
        logger.info(f"📦 Predict #{request_id}: Image size: {image_size_kb:.2f} KB")
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        logger.info(f"🖼️  Predict #{request_id}: Image: {image.size} {image.mode}")
        
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        logger.info(f"🤖 Predict #{request_id}: Running inference...")
        inference_start = time.time()
        
        # Run YOLO detection
        results = model.predict(
            img_array,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        
        inference_time = time.time() - inference_start
        logger.info(f"⚡ Predict #{request_id}: Inference done in {inference_time:.3f}s")
        
        # Get best detection
        best_detection = None
        best_confidence = 0
        detection_count = 0
        
        for r in results:
            boxes = r.boxes
            detection_count = len(boxes)
            logger.info(f"📊 Predict #{request_id}: Found {detection_count} objects")
            
            for idx, box in enumerate(boxes):
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                logger.info(f"   Object {idx+1}: {class_name} ({confidence:.2%})")
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = class_name
        
        processing_time = time.time() - start_time
        
        if best_detection:
            result_text = f"{best_detection}"
            logger.info(f"✅ Predict #{request_id}: SUCCESS - Best: {best_detection} ({best_confidence:.2%})")
        else:
            result_text = "No objects detected"
            logger.info(f"⚠️  Predict #{request_id}: No objects detected")
        
        logger.info(f"⏱️  Predict #{request_id}: Total time: {processing_time:.3f}s")
        logger.info(f"{'='*60}\n")
        
        request_counter['success'] += 1
        
        return jsonify({
            'success': True,
            'result': result_text,
            'confidence': best_confidence,
            'detections_count': detection_count,
            'processing_time': round(processing_time, 3),
            'request_id': request_id
        })
    
    except Exception as e:
        logger.error(f"❌ Predict #{request_id}: ERROR - {str(e)}")
        logger.error(f"{'='*60}\n")
        request_counter['failed'] += 1
        return jsonify({
            'success': False,
            'error': str(e),
            'request_id': request_id
        }), 500

if __name__ == '__main__':
    logger.info("\n" + "=" * 60)
    logger.info("🔥 YOLO Flask Backend Starting...")
    logger.info("=" * 60)
    logger.info(f"📦 Model: {MODEL_PATH}")
    logger.info(f"🎯 Classes: {list(model.names.values())}")
    logger.info(f"🌐 Server: http://0.0.0.0:5000")
    logger.info(f"📱 For mobile: http://YOUR_IP:5000")
    logger.info("=" * 60)
    logger.info("\n💡 Endpoints:")
    logger.info("   GET  /health          - Check server status")
    logger.info("   POST /predict         - Mobile app prediction (simplified)")
    logger.info("   POST /detect          - High quality detection")
    logger.info("   POST /detect-stream   - Fast real-time detection")
    logger.info("\n🚀 Server running with detailed logging...\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
