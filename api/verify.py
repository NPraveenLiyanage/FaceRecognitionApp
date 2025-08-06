import numpy as np
import tensorflow as tf
from http.server import BaseHTTPRequestHandler
import json

# Load your model (update path as needed)
model = tf.keras.models.load_model('../siamesemodelV2.h5')

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        # Example: expects 'img1' and 'img2' as base64 strings
        # You should add your own image preprocessing here
        # result = model.predict(...)
        result = {'success': True, 'message': 'API endpoint works!'}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
