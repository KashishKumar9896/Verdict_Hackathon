from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path to import YOLO
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Request counter for tracking
request_counter = {'total': 0, 'success': 0, 'failed': 0}

# Load YOLO model
logger.info("=" * 70)
logger.info("🚀 Starting YOLO Detection Server...")
logger.info("=" * 70)

try:
    from ultralytics import YOLO
    
    # Using the same model as test_model.py (continue_training_20epochs)
    MODEL_PATH = "../../Verdict_Hackathon/yolov8-project/runs/detect/continue_training_20epochs/weights/best.pt"
    
    logger.info(f"📂 Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    logger.info("✅ YOLO model loaded successfully!")
    logger.info(f"📊 Model classes: {list(model.names.values())}")
    logger.info(f"🔢 Number of classes: {len(model.names)}")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"❌ Failed to load YOLO model: {str(e)}")
    logger.error(f"   Please check the model path: {MODEL_PATH}")
    MODEL_LOADED = False
    model = None

logger.info("=" * 70)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from mobile app"""
    request_counter['total'] += 1
    request_id = request_counter['total']
    client_ip = request.remote_addr
    
    logger.info("\n" + "=" * 70)
    logger.info(f"📱 NEW REQUEST #{request_id} from {client_ip}")
    logger.info("=" * 70)
    
    if not MODEL_LOADED or model is None:
        logger.error(f"❌ Request #{request_id}: Model not loaded!")
        request_counter['failed'] += 1
        return jsonify({
            'success': False,
            'error': 'Model not loaded on server'
        }), 500
    
    try:
        import time
        start_time = time.time()
        
        # Get image from request
        data = request.json
        
        if not data:
            logger.error(f"❌ Request #{request_id}: No JSON data received")
            request_counter['failed'] += 1
            return jsonify({'success': False, 'error': 'No JSON data'}), 400
        
        img_base64 = data.get('image')
        
        if not img_base64:
            logger.error(f"❌ Request #{request_id}: No image field in JSON")
            request_counter['failed'] += 1
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        logger.info(f"✅ Request #{request_id}: Image data received")
        
        # Calculate base64 size
        image_size_kb = len(img_base64) / 1024
        logger.info(f"📦 Request #{request_id}: Image size: {image_size_kb:.2f} KB (base64)")
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if ',' in img_base64:
                img_base64 = img_base64.split(',')[1]
            
            img_data = base64.b64decode(img_base64)
            logger.info(f"🔓 Request #{request_id}: Base64 decoded successfully")
            
            # Convert to PIL Image
            image = Image.open(BytesIO(img_data))
            logger.info(f"🖼️  Request #{request_id}: Image opened - Size: {image.size}, Mode: {image.mode}")
            
            # Resize image if it's too large (to match training data size ~640px)
            original_size = image.size
            max_size = 1280  # Resize large images for better performance
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"🔄 Request #{request_id}: Resized from {original_size} to {image.size}")
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert RGB to BGR for YOLO
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            logger.info(f"🔄 Request #{request_id}: Image converted - Shape: {img_array.shape}")
            
        except Exception as decode_error:
            logger.error(f"❌ Request #{request_id}: Image decode failed - {str(decode_error)}")
            request_counter['failed'] += 1
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400
        
        # Run YOLO inference
        logger.info(f"🤖 Request #{request_id}: Starting YOLO inference...")
        inference_start = time.time()
        
        try:
            results = model.predict(
                img_array,
                conf=0.25,  # Confidence threshold (same as test_model.py)
                iou=0.45,   # IoU threshold (same as test_model.py)
                verbose=False,
                imgsz=640   # Image size for inference (standard YOLO size)
            )
            
            inference_time = time.time() - inference_start
            logger.info(f"⚡ Request #{request_id}: Inference completed in {inference_time:.3f}s")
            
        except Exception as inference_error:
            logger.error(f"❌ Request #{request_id}: Inference failed - {str(inference_error)}")
            request_counter['failed'] += 1
            return jsonify({'success': False, 'error': f'Inference error: {str(inference_error)}'}), 500
        
        # Process results - return ALL detections with bounding boxes
        best_detection = None
        best_confidence = 0
        detection_count = 0
        all_detections = []
        
        # Get original image dimensions for coordinate normalization
        img_height, img_width = img_array.shape[:2]
        
        for r in results:
            boxes = r.boxes
            detection_count = len(boxes)
            logger.info(f"📊 Request #{request_id}: Found {detection_count} objects")
            
            for idx, box in enumerate(boxes):
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Normalize coordinates to 0-1 range for mobile display
                normalized_box = {
                    'x1': x1 / img_width,
                    'y1': y1 / img_height,
                    'x2': x2 / img_width,
                    'y2': y2 / img_height
                }
                
                logger.info(f"   Detection #{idx+1}: {class_name} ({confidence:.2%}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                
                all_detections.append({
                    'class': class_name,
                    'confidence': round(confidence, 3),
                    'box': normalized_box,  # Normalized coordinates
                    'box_pixels': {  # Pixel coordinates for reference
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2)
                    }
                })
                
                # Track best detection
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = class_name
        
        # Prepare response
        total_time = time.time() - start_time
        
        if best_detection:
            result = f"{best_detection}"
            if detection_count > 1:
                result = f"{detection_count} objects: {best_detection} + more"
            logger.info(f"✅ Request #{request_id}: SUCCESS - Best detection: {best_detection} ({best_confidence:.2%})")
        else:
            result = "No objects detected"
            logger.info(f"⚠️  Request #{request_id}: No objects detected above threshold")
        
        logger.info(f"⏱️  Request #{request_id}: Total processing time: {total_time:.3f}s")
        logger.info(f"   - Inference: {inference_time:.3f}s")
        logger.info(f"   - Overhead: {(total_time - inference_time):.3f}s")
        logger.info("=" * 70 + "\n")
        
        request_counter['success'] += 1
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(best_confidence, 3),
            'detections_count': detection_count,
            'all_detections': all_detections,
            'processing_time': round(total_time, 3),
            'inference_time': round(inference_time, 3),
            'request_id': request_id,
            'image_info': {
                'size': image.size,
                'mode': image.mode
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Request #{request_id}: Unexpected error - {str(e)}")
        logger.exception("Full traceback:")
        logger.info("=" * 70 + "\n")
        request_counter['failed'] += 1
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with detailed status"""
    logger.info(f"🏥 Health check from {request.remote_addr}")
    
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'model_not_loaded',
        'model_loaded': MODEL_LOADED,
        'model_classes': list(model.names.values()) if MODEL_LOADED else [],
        'total_requests': request_counter['total'],
        'successful_requests': request_counter['success'],
        'failed_requests': request_counter['failed'],
        'success_rate': f"{(request_counter['success'] / max(request_counter['total'], 1) * 100):.1f}%",
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    }), 200

if __name__ == '__main__':
    logger.info("\n" + "=" * 70)
    logger.info("🌐 Starting Flask Server...")
    logger.info("=" * 70)
    logger.info(f"📍 Server Address: http://0.0.0.0:5000")
    logger.info(f"📱 Mobile Access: http://YOUR_IP_ADDRESS:5000")
    logger.info(f"🔧 Model Status: {'✅ Loaded' if MODEL_LOADED else '❌ Not Loaded'}")
    logger.info("=" * 70)
    logger.info("\n💡 Available Endpoints:")
    logger.info("   GET  /health   - Check server and model status")
    logger.info("   POST /predict  - Send image for object detection")
    logger.info("\n🚀 Server is running with detailed logging enabled...")
    logger.info("   Watch this console for real-time request logs!\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)