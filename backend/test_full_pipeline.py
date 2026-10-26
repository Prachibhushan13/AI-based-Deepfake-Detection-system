import sys
sys.path.insert(0, '.')
from app.services.video_service import save_upload, compress_video, extract_frames
from app.ml.inference import predict_video
import numpy as np
import cv2
import io
from fastapi import UploadFile

print('=== Testing full video processing pipeline ===')

# Create a test video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (224, 224))
for i in range(60):
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[:, :, 0] = (i * 4) % 256
    frame[:, :, 1] = (i * 8) % 256
    frame[:, :, 2] = (i * 12) % 256
    out.write(frame)
out.release()

# Create a mock UploadFile
class MockUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.file = io.BytesIO(content)

with open('test_video.mp4', 'rb') as f:
    content = f.read()

mock_file = MockUploadFile('test_video.mp4', content)

# Test save_upload
saved_path = save_upload(mock_file)
print(f'1. Saved to: {saved_path}')

# Test compress_video
compressed_path = compress_video(saved_path)
print(f'2. Compressed to: {compressed_path}')

# Test extract_frames
frames = extract_frames(compressed_path)
print(f'3. Extracted {len(frames)} frames')

# Test predict_video
result = predict_video(frames, 'test_video.mp4')
print(f'4. Prediction: {result["result"]}')
print(f'5. Fake probability: {result["fakeProbability"]}')
print(f'7. Model mode: {result["modelMode"]}')

# Cleanup
import os
os.remove('test_video.mp4')
if saved_path.exists():
    os.remove(saved_path)
if compressed_path.exists() and compressed_path != saved_path:
    os.remove(compressed_path)

print('\n=== Full pipeline test passed ===')