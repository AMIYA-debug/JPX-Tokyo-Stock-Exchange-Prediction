from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime
import traceback

app = Flask(__name__)

# Load saved models and scalers
try:
    model = tf.keras.models.load_model('tuned_regression_model.keras')
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))
    
    # Load all 5 individual encoders
    oh1 = pickle.load(open('onehot_encoder.pkl', 'rb'))      # NewMarketSegment
    oh2 = pickle.load(open('onehot_encoder_2.pkl', 'rb'))    # Section/Products
    oh3 = pickle.load(open('onehot_encoder_3.pkl', 'rb'))    # 33SectorCode
    oh4 = pickle.load(open('onehot_encoder_4.pkl', 'rb'))    # 17SectorCode
    oh5 = pickle.load(open('onehot_encoder_5.pkl', 'rb'))    # NewIndexSeriesSizeCode
    
    # Load feature columns for alignment
    feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
    
    print("✓ Models, scalers, and encoders loaded successfully!")
    print(f"✓ Loaded {len(feature_columns)} feature columns")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'Universe0': int(request.form.get('universe0', 0)),
                'Open': float(request.form.get('open')),
                'High': float(request.form.get('high')),
                'Low': float(request.form.get('low')),
                'Volume': float(request.form.get('volume')),
                'NewMarketSegment': request.form.get('market_segment'),
                'Section/Products': request.form.get('section_products'),
                '33SectorCode': int(request.form.get('sector_33')),
                '17SectorCode': int(request.form.get('sector_17')),
                'NewIndexSeriesSizeCode': int(request.form.get('index_size_code')),
                'TradeDate': request.form.get('trade_date')
            }
            
            # Validate inputs
            if not all([data['Open'], data['High'], data['Low'], data['Volume'], 
                       data['NewMarketSegment'], data['Section/Products'], data['TradeDate']]):
                raise ValueError("All fields are required")
            
            # Process trade date
            trade_date = pd.to_datetime(data['TradeDate'])
            
            # Create base DataFrame with numeric and date features (matching training order)
            input_df = pd.DataFrame({
                'Universe0': [data['Universe0']],
                'Open': [data['Open']],
                'High': [data['High']],
                'Low': [data['Low']],
                'Volume': [data['Volume']],
                'Year': [trade_date.year],
                'Month': [trade_date.month],
                'Day': [trade_date.day],
                'DayOfWeek': [trade_date.dayofweek],
                'Quarter': [trade_date.quarter],
                'IsMonthStart': [int(trade_date.is_month_start)],
                'IsMonthEnd': [int(trade_date.is_month_end)],
                'IsQuarterEnd': [int(trade_date.is_quarter_end)],
                'IsYearEnd': [int(trade_date.is_year_end)]
            })
            
            # Apply one-hot encoding using the SAME fitted encoders (oh1 through oh5)
            # Encoder 1: NewMarketSegment
            market_encoded = oh1.transform([[data['NewMarketSegment']]])
            market_encoded_df = pd.DataFrame(
                market_encoded, 
                columns=oh1.get_feature_names_out(['NewMarketSegment'])
            )
            
            # Encoder 2: Section/Products
            products_encoded = oh2.transform([[data['Section/Products']]])
            products_encoded_df = pd.DataFrame(
                products_encoded, 
                columns=oh2.get_feature_names_out(['Section/Products'])
            )
            
            # Encoder 3: 33SectorCode
            code33_encoded = oh3.transform([[data['33SectorCode']]])
            code33_encoded_df = pd.DataFrame(
                code33_encoded, 
                columns=oh3.get_feature_names_out(['33SectorCode'])
            )
            
            # Encoder 4: 17SectorCode
            code17_encoded = oh4.transform([[data['17SectorCode']]])
            code17_encoded_df = pd.DataFrame(
                code17_encoded, 
                columns=oh4.get_feature_names_out(['17SectorCode'])
            )
            
            # Encoder 5: NewIndexSeriesSizeCode
            code_new_index_encoded = oh5.transform([[data['NewIndexSeriesSizeCode']]])
            code_new_index_encoded_df = pd.DataFrame(
                code_new_index_encoded, 
                columns=oh5.get_feature_names_out(['NewIndexSeriesSizeCode'])
            )
            
            # Concatenate all features in the same order as training
            input_df = pd.concat([
                input_df, 
                market_encoded_df, 
                products_encoded_df, 
                code33_encoded_df, 
                code17_encoded_df, 
                code_new_index_encoded_df
            ], axis=1)
            
            # Align columns with training data
            # Add missing columns with 0s (for categories not present in this prediction)
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training order exactly
            input_df = input_df[feature_columns]
            
            print(f"✓ Input shape: {input_df.shape}")
            print(f"✓ Expected shape: ({len(feature_columns)},)")
            
            # Scale the input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction_scaled = model.predict(input_scaled, verbose=0)
            prediction = scaler_y.inverse_transform(prediction_scaled)
            
            predicted_price = float(prediction[0][0])
            
            print(f"✓ Prediction successful: ¥{predicted_price:.2f}")
            
            return render_template('predict.html', 
                                 prediction=predicted_price,
                                 input_data=data)
        
        except ValueError as ve:
            error_message = f"Validation Error: {str(ve)}"
            print(f"✗ {error_message}")
            return render_template('predict.html', error=error_message)
        
        except Exception as e:
            error_message = f"Prediction Error: {str(e)}"
            print(f"✗ {error_message}")
            print(traceback.format_exc())
            return render_template('predict.html', error=error_message)
    
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
