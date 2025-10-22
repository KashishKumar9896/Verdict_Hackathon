# YOLO Detection Server - Logging Guide

## 🚀 Quick Start

### 1. Start the Server

```bash
cd "d:\hackathon\expo app\CodeGeass"
python server.py
```

### 2. What You'll See

The server will display comprehensive logs showing:

#### On Startup:
```
======================================================================
🚀 Starting YOLO Detection Server...
======================================================================
📂 Loading model from: ../../Verdict_Hackathon/yolov8-project/...
✅ YOLO model loaded successfully!
📊 Model classes: ['OxygenTank', 'NitrogenTank', 'FirstAidBox', ...]
🔢 Number of classes: 7
======================================================================
🌐 Starting Flask Server...
📍 Server Address: http://0.0.0.0:5000
📱 Mobile Access: http://YOUR_IP_ADDRESS:5000
🔧 Model Status: ✅ Loaded
======================================================================
```

#### On Each Request:
```
======================================================================
📱 NEW REQUEST #1 from 192.168.1.100
======================================================================
✅ Request #1: Image data received
📦 Request #1: Image size: 145.23 KB (base64)
🔓 Request #1: Base64 decoded successfully
🖼️  Request #1: Image opened - Size: (1080, 1920), Mode: RGB
🔄 Request #1: Image converted - Shape: (1920, 1080, 3)
🤖 Request #1: Starting YOLO inference...
⚡ Request #1: Inference completed in 0.234s
📊 Request #1: Found 2 objects
   Detection #1: FireExtinguisher (89.45%) at [123, 456, 234, 567]
   Detection #2: FirstAidBox (76.23%) at [345, 678, 456, 789]
✅ Request #1: SUCCESS - Best detection: FireExtinguisher (89.45%)
⏱️  Request #1: Total processing time: 0.456s
   - Inference: 0.234s
   - Overhead: 0.222s
======================================================================
```

## 📱 Mobile App Logs

### In Expo Terminal (Metro Bundler)

When you toggle detection on/off, you'll see:

```
🔍 Checking server connection...
   Server URL: http://172.17.152.26:5000/health
   Response status: 200
✅ Server connected successfully!
   Model loaded: true
   Classes available: OxygenTank, NitrogenTank, FirstAidBox, ...
   Total requests: 15
   Success: 14 | Failed: 1
```

### During Image Capture:

```
============================================================
📸 CAPTURING IMAGE FOR DETECTION
============================================================
📷 Taking picture...
✅ Picture captured in 234ms
   Size: 1080x1920
📦 Base64 size: 145.23 KB
🚀 Sending to server: http://172.17.152.26:5000/predict
📤 Upload completed in 456ms
   Response status: 200 OK
📥 Response received from server:
   Success: true
   Result: FireExtinguisher
   Confidence: 89.5%
✅ Detection successful!
⏱️  Total time: 690ms
============================================================
```

## 🔧 Troubleshooting

### If Model Doesn't Load:

1. Check the model path in `server.py`:
   ```python
   MODEL_PATH = "../../Verdict_Hackathon/yolov8-project/runs/detect/continue_training_20more_v2/weights/best.pt"
   ```

2. Update it to your actual best model path

3. Look for this in logs:
   ```
   ❌ Failed to load YOLO model: [Errno 2] No such file or directory
   ```

### If App Can't Connect:

1. Check your IP address:
   ```bash
   ipconfig  # Windows
   ifconfig  # Mac/Linux
   ```

2. Update `SERVER_IP` in `camera.tsx` and `index.tsx`:
   ```typescript
   const SERVER_IP = 'YOUR_IP_HERE';  // e.g., '192.168.1.100'
   ```

3. Make sure both devices are on the same network

4. Check firewall allows port 5000

## 📊 Server Statistics

Visit `http://YOUR_IP:5000/health` in browser to see:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_classes": ["OxygenTank", "NitrogenTank", ...],
  "total_requests": 42,
  "successful_requests": 40,
  "failed_requests": 2,
  "success_rate": "95.2%",
  "server_time": "2025-10-15 14:30:25"
}
```

## 🎯 What Each Log Means

| Icon | Meaning |
|------|---------|
| 🚀 | Server starting / Sending request |
| ✅ | Success |
| ❌ | Error |
| 📸 | Image capture |
| 📦 | Data size information |
| 🔓 | Decoding operation |
| 🖼️  | Image information |
| 🤖 | AI model operation |
| ⚡ | Performance metric |
| 📊 | Statistics |
| ⏱️  | Timing information |
| 🏥 | Health check |
| 📱 | Mobile app request |

## 💡 Tips

1. **Performance Monitoring**: Watch the inference time - should be < 1 second
2. **Network Issues**: If upload time > 2 seconds, check network connection
3. **Model Issues**: If inference fails, model might not be loaded correctly
4. **Image Size**: Larger images = slower upload. Current quality: 0.6 (60%)

## 🔍 Debug Mode

To see even more details, you can:

1. In `server.py`, change:
   ```python
   app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
   ```

2. In React Native, open Chrome DevTools:
   - Press `j` in Metro bundler terminal
   - Open Chrome to `chrome://inspect`
   - View all console.log messages

---

**Need Help?**
- Check server logs for error messages
- Check Expo Metro bundler logs for app errors
- Verify IP address matches between server and app
- Ensure model file exists at specified path
