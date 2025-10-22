import React, { useRef, useState, useEffect } from 'react';
import { View, Text, ActivityIndicator, StyleSheet, Switch, TouchableOpacity, Dimensions, ScrollView } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

const SERVER_IP = '172.17.152.26'; // Change to your Python server IP
const SERVER_PORT = 5000;
const SERVER_URL = `http://${SERVER_IP}:${SERVER_PORT}`;
const DETECTION_INTERVAL = 400; // Send frame every 1 second

interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface Detection {
  class: string;
  confidence: number;
  box?: DetectionBox;
  box_pixels?: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

interface PredictionResponse {
  success: boolean;
  result?: string;
  confidence?: number;
  error?: string;
  all_detections?: Detection[];
  detections_count?: number;
}

interface HealthResponse {
  status: string;
}

export default function App(): React.ReactElement {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [result, setResult] = useState<string>('Waiting for detection...');
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [serverStatus, setServerStatus] = useState<string>('Checking...');
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    checkServerConnection();
  }, []);

  useEffect(() => {
    if (isDetecting) {
      startLiveDetection();
    } else {
      stopLiveDetection();
    }

    return () => {
      stopLiveDetection();
    };
  }, [isDetecting]);

  const checkServerConnection = async (): Promise<void> => {
    try {
      const response = await fetch(`${SERVER_URL}/health`);
      
      if (response.ok) {
        const data = await response.json();
        setServerStatus('Connected');
        console.log('✅ Server connected');
      }
    } catch (error) {
      setServerStatus('Disconnected');
      console.error('❌ Server connection failed');
    }
  };

  const startLiveDetection = (): void => {
    intervalRef.current = setInterval(() => {
      captureAndPredict();
    }, DETECTION_INTERVAL);
  };

  const stopLiveDetection = (): void => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const captureAndPredict = async (): Promise<void> => {
    if (!cameraRef.current || loading) return;

    setLoading(true);
    try {
      // Capture photo
      const photoData = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.8,  // Increased quality for better detection
        skipProcessing: true,
        exif: false,
      });
      
      if (!photoData.base64) {
        console.error('❌ No base64 data in captured photo');
        setLoading(false);
        return;
      }

      // Send to Python server
      const response = await fetch(`${SERVER_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: photoData.base64,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = (await response.json()) as PredictionResponse;

      if (data.success) {
        setResult(data.result || 'Unknown');
        setConfidence(data.confidence || null);
        setDetections(data.all_detections || []);
        
        // Log all detected objects line by line
        if (data.all_detections && data.all_detections.length > 0) {
          console.log('\n' + '🎯 DETECTED OBJECTS:');
          console.log('━'.repeat(50));
          data.all_detections.forEach((det, idx) => {
            console.log(`${idx + 1}. ${det.class.padEnd(20)} | Confidence: ${(det.confidence * 100).toFixed(1)}%`);
          });
          console.log('━'.repeat(50));
          console.log(`📊 Total: ${data.all_detections.length} objects detected\n`);
        } else {
          console.log('⚠️  No objects detected');
        }
      } else {
        setResult(`Error: ${data.error}`);
        setConfidence(null);
        setDetections([]);
        console.error(`❌ Error: ${data.error}`);
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setResult(`Error: ${errorMessage}`);
      setConfidence(null);
      setDetections([]);
      console.error(`❌ Error: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  const handleManualCapture = (): void => {
    captureAndPredict();
  };

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>Camera permission required</Text>
        <Text style={styles.permissionButton} onPress={requestPermission}>
          Grant Permission
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera View */}
      <CameraView ref={cameraRef} style={styles.camera} facing="back" />

      {/* Detection Boxes Overlay */}
      <View style={styles.detectionOverlay} pointerEvents="none">
        {detections.map((detection, index) => {
          if (!detection.box) return null;
          
          const screenWidth = Dimensions.get('window').width;
          const screenHeight = Dimensions.get('window').height;
          
          // Convert normalized coordinates to screen coordinates
          const left = detection.box.x1 * screenWidth;
          const top = detection.box.y1 * screenHeight;
          const width = (detection.box.x2 - detection.box.x1) * screenWidth;
          const height = (detection.box.y2 - detection.box.y1) * screenHeight;
          
          // Color based on class
          const colors = ['#00FF00', '#FF00FF', '#00FFFF', '#FFFF00', '#FF8800', '#FF0088', '#88FF00'];
          const boxColor = colors[index % colors.length];
          
          return (
            <View
              key={`detection-${index}`}
              style={[
                styles.detectionBox,
                {
                  left,
                  top,
                  width,
                  height,
                  borderColor: boxColor,
                },
              ]}
            >
              <View style={[styles.detectionLabel, { backgroundColor: boxColor }]}>
                <Text style={styles.detectionLabelText}>
                  {detection.class} {(detection.confidence * 100).toFixed(0)}%
                </Text>
              </View>
            </View>
          );
        })}
      </View>

      {/* Overlay */}
      <View style={styles.overlay}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Live Detection</Text>
          <View style={[styles.statusBadge, serverStatus === 'Connected' ? styles.statusConnected : styles.statusDisconnected]}>
            <View style={[styles.statusDot, serverStatus === 'Connected' ? styles.dotConnected : styles.dotDisconnected]} />
            <Text style={styles.statusText}>{serverStatus}</Text>
          </View>
        </View>

        {/* Center Focus Area */}
        <View style={styles.centerArea}>
          <View style={styles.focusBox}>
            <View style={[styles.corner, styles.topLeft]} />
            <View style={[styles.corner, styles.topRight]} />
            <View style={[styles.corner, styles.bottomLeft]} />
            <View style={[styles.corner, styles.bottomRight]} />
          </View>
        </View>

        {/* Bottom Section */}
        <View style={styles.bottomSection}>
          {/* Detection Toggle */}
          <View style={styles.toggleContainer}>
            <Text style={styles.toggleLabel}>
              {isDetecting ? 'Detection Active' : 'Detection Paused'}
            </Text>
            <Switch
              value={isDetecting}
              onValueChange={setIsDetecting}
              trackColor={{ false: '#767577', true: '#81b0ff' }}
              thumbColor={isDetecting ? '#007AFF' : '#f4f3f4'}
            />
          </View>

          {/* Manual Capture Button */}
          <TouchableOpacity
            style={[styles.captureButton, loading && styles.captureButtonDisabled]}
            onPress={handleManualCapture}
            disabled={loading}
            activeOpacity={0.8}
          >
            <View style={styles.captureButtonInner}>
              <Text style={styles.captureButtonIcon}>📸</Text>
              <Text style={styles.captureButtonText}>
                {loading ? 'Processing...' : 'Capture Photo'}
              </Text>
            </View>
          </TouchableOpacity>

          {/* Results Display */}
          <View style={styles.resultsCard}>
            {loading && (
              <View style={styles.loadingRow}>
                <ActivityIndicator size="small" color="#007AFF" />
                <Text style={styles.loadingText}>Analyzing...</Text>
              </View>
            )}
            
            <View style={styles.resultRow}>
              <Text style={styles.resultLabel}>Detection:</Text>
              <Text style={styles.resultValue}>{result}</Text>
            </View>

            {confidence !== null && (
              <View style={styles.confidenceContainer}>
                <Text style={styles.confidenceLabel}>Best Confidence</Text>
                <View style={styles.confidenceBar}>
                  <View style={[styles.confidenceFill, { width: `${confidence * 100}%` }]} />
                </View>
                <Text style={styles.confidenceText}>{(confidence * 100).toFixed(1)}%</Text>
              </View>
            )}

            {/* Show all detections */}
            {detections.length > 0 && (
              <View style={styles.allDetectionsContainer}>
                <Text style={styles.allDetectionsLabel}>All Objects ({detections.length}):</Text>
                <ScrollView 
                  style={styles.detectionScrollView}
                  showsVerticalScrollIndicator={true}
                  nestedScrollEnabled={true}
                >
                  {detections.map((det, idx) => (
                    <View key={`det-${idx}`} style={styles.detectionItem}>
                      <Text style={styles.detectionItemText}>
                        {idx + 1}. {det.class}
                      </Text>
                      <Text style={styles.detectionItemConfidence}>
                        {(det.confidence * 100).toFixed(1)}%
                      </Text>
                    </View>
                  ))}
                </ScrollView>
              </View>
            )}
          </View>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  detectionOverlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 1,
  },
  detectionBox: {
    position: 'absolute',
    borderWidth: 3,
    borderStyle: 'solid',
  },
  detectionLabel: {
    position: 'absolute',
    top: -25,
    left: 0,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  detectionLabelText: {
    color: '#000',
    fontSize: 12,
    fontWeight: 'bold',
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
    zIndex: 2,
  },
  header: {
    paddingTop: 50,
    paddingHorizontal: 20,
    paddingBottom: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    gap: 6,
  },
  statusConnected: {
    backgroundColor: 'rgba(52, 199, 89, 0.2)',
  },
  statusDisconnected: {
    backgroundColor: 'rgba(255, 59, 48, 0.2)',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  dotConnected: {
    backgroundColor: '#34C759',
  },
  dotDisconnected: {
    backgroundColor: '#FF3B30',
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
  },
  centerArea: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  focusBox: {
    width: 280,
    height: 280,
    position: 'relative',
  },
  corner: {
    position: 'absolute',
    width: 40,
    height: 40,
    borderColor: '#007AFF',
    borderWidth: 3,
  },
  topLeft: {
    top: 0,
    left: 0,
    borderBottomWidth: 0,
    borderRightWidth: 0,
    borderTopLeftRadius: 8,
  },
  topRight: {
    top: 0,
    right: 0,
    borderBottomWidth: 0,
    borderLeftWidth: 0,
    borderTopRightRadius: 8,
  },
  bottomLeft: {
    bottom: 0,
    left: 0,
    borderTopWidth: 0,
    borderRightWidth: 0,
    borderBottomLeftRadius: 8,
  },
  bottomRight: {
    bottom: 0,
    right: 0,
    borderTopWidth: 0,
    borderLeftWidth: 0,
    borderBottomRightRadius: 8,
  },
  bottomSection: {
    paddingHorizontal: 20,
    paddingBottom: 40,
    gap: 16,
  },
  toggleContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderRadius: 16,
  },
  toggleLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  captureButton: {
    backgroundColor: '#007AFF',
    borderRadius: 16,
    paddingVertical: 18,
    paddingHorizontal: 24,
    shadowColor: '#007AFF',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  captureButtonDisabled: {
    backgroundColor: '#4A4A4A',
    shadowOpacity: 0,
  },
  captureButtonInner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  captureButtonIcon: {
    fontSize: 24,
  },
  captureButtonText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
    textAlign: 'center',
  },
  resultsCard: {
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderRadius: 16,
    padding: 20,
    gap: 12,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  loadingText: {
    fontSize: 14,
    color: '#007AFF',
    fontWeight: '500',
  },
  resultRow: {
    gap: 4,
  },
  resultLabel: {
    fontSize: 12,
    color: '#999',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  resultValue: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
  },
  confidenceContainer: {
    gap: 6,
    marginTop: 8,
  },
  confidenceLabel: {
    fontSize: 12,
    color: '#999',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  confidenceBar: {
    height: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    backgroundColor: '#007AFF',
  },
  confidenceText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '600',
    alignSelf: 'flex-end',
  },
  allDetectionsContainer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.2)',
  },
  allDetectionsLabel: {
    fontSize: 12,
    color: '#999',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
  },
  detectionScrollView: {
    maxHeight: 200, // Limit height to show ~8 items, rest are scrollable
  },
  detectionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 6,
  },
  detectionItemText: {
    fontSize: 14,
    color: '#fff',
    fontWeight: '500',
  },
  detectionItemConfidence: {
    fontSize: 14,
    color: '#007AFF',
    fontWeight: '600',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  permissionText: {
    fontSize: 16,
    color: '#fff',
    marginBottom: 16,
    fontWeight: '500',
  },
  permissionButton: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '600',
  },
});