from flask import Flask, send_file, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

DATA_DIR = "/Users/kshitijnavale/Desktop/sensor data/data"

@app.route('/api/data/<filename>')
def serve_csv(filename):
    try:
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='text/csv')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 CSV Server starting...")
    print(f"📁 Serving files from: {DATA_DIR}")
    print(f"🌐 Server: http://localhost:5002")
    app.run(debug=True, port=5002)
