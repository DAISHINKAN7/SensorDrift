import os
import sys

# Set environment before importing anything
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Now run the app
if __name__ == '__main__':
    from app import app
    app.run(debug=False, host='0.0.0.0', port=5001)