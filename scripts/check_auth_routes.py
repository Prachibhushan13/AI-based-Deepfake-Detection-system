import sys
sys.path.insert(0, r'D:\AI-based-Deepfake-Detection-system-main\backend')
from app.main import app
for route in app.routes:
    if hasattr(route, 'path') and 'auth' in route.path:
        print('Route:', route.path, 'methods:', getattr(route, 'methods', None))