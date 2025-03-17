from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

app = Flask(__name__)

# Initialize preprocessing objects
le_grade = LabelEncoder()
scaler = MinMaxScaler()

@app.route('/')
def home():
    return jsonify({"status": "API is running", "message": "Use /predict endpoint with grade and time_taken parameters"})

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get parameters safely
        grade = request.args.get('grade')
        time_taken = request.args.get('time_taken')
        
        # Validate inputs
        if not grade or not time_taken:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Convert to appropriate types
        current_grade = int(grade)
        time_taken = float(time_taken)
        
        # Input validation
        if current_grade < 1 or current_grade > 10:
            return jsonify({'error': 'Grade must be between 1 and 10'}), 400
        if time_taken < 0:
            return jsonify({'error': 'Time taken cannot be negative'}), 400
        
        # Prepare data for prediction
        data = {
            'current_grade': [current_grade],
            'time_taken': [time_taken]
        }
        df = pd.DataFrame(data)
        
        # Encode grade (fit on possible grades 1-10)
        le_grade.fit(range(1, 11))
        df['encoded_grade'] = le_grade.transform([current_grade])
        
        # Scale features (simple min-max scaling)
        df['scaled_time'] = time_taken / 180  # Assuming max time is 180 seconds
        
        # Determine grade adjustment based on time taken
        if time_taken < 60:
            adjusted_grade = min(current_grade + 1, 10)  # Cap at grade 10
        elif time_taken > 90:
            adjusted_grade = max(current_grade - 1, 1)   # Minimum grade 1
        else:
            adjusted_grade = current_grade
            
        return jsonify({
            'status': 'success',
            'adjusted_grade': adjusted_grade,
            'adjustment': adjusted_grade - current_grade,
            'input_data': {
                'original_grade': current_grade,
                'time_taken': time_taken
            }
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# This is important for Vercel
app.debug = True
