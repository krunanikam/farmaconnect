import smtplib
import random
import sqlite3
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import logging
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import traceback

# Suppress warnings
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "farmaaconnect@gmail.com"
EMAIL_PASSWORD = "sjahxbemfppchjmc"
FARMER_DB = "farmers.db"
VENDOR_DB = "vendors.db"
OTP_EXPIRY_MINUTES = 5
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'models/lstm_model.h5'
SCALER_PATH = 'models/scaler.pkl'
DATA_FILES = ['data/20.csv', 'data/21.csv', 'data/22.csv', 'data/23.csv', 'data/24.csv']
TIME_STEP = 10

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Cache for storing loaded data to improve performance
DATA_CACHE = {}
otp_storage = {}

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_folder():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

# Database Setup for Farmers
def init_farmer_db():
    conn = None
    try:
        conn = sqlite3.connect(FARMER_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS farmers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                commodity TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                address TEXT NOT NULL,
                mobile TEXT NOT NULL,
                email TEXT NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_farmers_commodity ON farmers(commodity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_farmers_address ON farmers(address)')
        
        conn.commit()
        logger.info("Farmer database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Farmer database initialization failed: {e}")
        raise
    finally:
        if conn:
            conn.close()

# Database Setup for Vendors
def init_vendor_db():
    conn = None
    try:
        conn = sqlite3.connect(VENDOR_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vendors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                market_name TEXT NOT NULL,
                commodity TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                email TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vendors_commodity ON vendors(commodity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vendors_market ON vendors(market_name)')
        
        conn.commit()
        logger.info("Vendor database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Vendor database initialization failed: {e}")
        raise
    finally:
        if conn:
            conn.close()

# Initialize both databases
init_farmer_db()
init_vendor_db()

# Email Service
def send_email(to_email, otp):
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        
        subject = "Your OTP Code"
        body = f"""
        Your OTP for verification is: {otp}
        
        This OTP is valid for {OTP_EXPIRY_MINUTES} minutes.
        
        If you didn't request this, please ignore this email.
        """
        message = f"Subject: {subject}\n\n{body}"
        server.sendmail(EMAIL_ADDRESS, to_email, message)
        server.quit()
        logger.info(f"OTP sent to {to_email}")
        return True
    except smtplib.SMTPException as e:
        logger.error(f"SMTP Error: {e}")
        return False

# Helper function to clean expired OTPs
def clean_expired_otps():
    now = datetime.now()
    expired_keys = [key for key, data in otp_storage.items() if data['expiry'] < now]
    for key in expired_keys:
        del otp_storage[key]
    if expired_keys:
        logger.info(f"Cleaned {len(expired_keys)} expired OTPs")

# ====================== COMMON ROUTES ======================
@app.route('/')
def home():
    return render_template('market_trend.html')

@app.route('/send-otp', methods=['POST'])
def send_otp():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data received"}), 400

        email = data.get("email", "").strip().lower()
        if not email:
            return jsonify({"success": False, "message": "Valid email is required"}), 400

        clean_expired_otps()

        otp = str(random.randint(100000, 999999))
        expiry = datetime.now() + timedelta(minutes=OTP_EXPIRY_MINUTES)
        
        otp_storage[email] = {
            'otp': otp,
            'expiry': expiry,
            'verified': False
        }

        if send_email(email, otp):
            return jsonify({
                "success": True,
                "message": f"OTP sent successfully. Valid for {OTP_EXPIRY_MINUTES} minutes."
            })
        return jsonify({"success": False, "message": "Failed to send OTP"}), 500

    except Exception as e:
        logger.error(f"Error in send-otp: {str(e)}")
        return jsonify({"success": False, "message": "Internal server error"}), 500

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data received"}), 400

        email = data.get("email", "").strip().lower()
        otp = data.get("otp", "").strip()

        if not email or not otp:
            return jsonify({"success": False, "message": "Email and OTP are required"}), 400

        clean_expired_otps()

        stored_data = otp_storage.get(email)
        if not stored_data:
            return jsonify({"success": False, "message": "OTP not found or expired"}), 404

        if stored_data['otp'] == otp:
            stored_data['verified'] = True
            return jsonify({"success": True, "message": "OTP verified"})
        
        return jsonify({"success": False, "message": "Invalid OTP"}), 400

    except Exception as e:
        logger.error(f"Error in verify-otp: {str(e)}")
        return jsonify({"success": False, "message": "Internal server error"}), 500

# ====================== FARMER ROUTES ======================
@app.route('/store-commodity', methods=['POST'])
def store_commodity():
    conn = None
    try:
        ensure_upload_folder()
        
        if 'photo' in request.files:
            file = request.files['photo']
            if file.filename == '':
                return jsonify({"success": False, "message": "No selected file"}), 400
                
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                image_path = file_path
            else:
                return jsonify({"success": False, "message": "Invalid file type"}), 400
        else:
            image_path = None

        if request.form:
            form_data = request.form
        else:
            form_data = request.get_json() or {}
        
        email = form_data.get("email", "").strip().lower()
        if not email:
            return jsonify({"success": False, "message": "Valid email is required"}), 400

        stored_data = otp_storage.get(email)
        if not stored_data or not stored_data.get('verified'):
            return jsonify({"success": False, "message": "Email not verified"}), 403

        fields = {
            "name": form_data.get("farmerName", "").strip(),
            "commodity": form_data.get("commodity", "").strip(),
            "quantity": form_data.get("quantity"),
            "price": form_data.get("price"),
            "address": form_data.get("address", "").strip(),
            "mobile": form_data.get("phone", "").strip(),
            "email": email,
            "image_path": image_path
        }

        if not all(value for key, value in fields.items() if key != 'image_path'):
            return jsonify({"success": False, "message": "All required fields are missing"}), 400

        try:
            fields["quantity"] = int(fields["quantity"])
            fields["price"] = float(fields["price"])
            if fields["quantity"] <= 0 or fields["price"] <= 0:
                raise ValueError("Values must be positive")
        except (ValueError, TypeError) as e:
            return jsonify({"success": False, "message": "Invalid quantity or price"}), 400

        conn = sqlite3.connect(FARMER_DB)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO farmers (name, commodity, quantity, price, address, mobile, email, image_path) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fields["name"], fields["commodity"], fields["quantity"],
            fields["price"], fields["address"], fields["mobile"], 
            fields["email"], fields["image_path"]
        ))
        conn.commit()
        
        del otp_storage[email]
        
        response_data = {
            "success": True,
            "message": "Farmer data stored successfully",
            "id": cursor.lastrowid
        }
        
        if image_path:
            filename = os.path.basename(image_path)
            response_data["image_url"] = f"/uploads/{filename}"
            
        return jsonify(response_data)

    except sqlite3.Error as e:
        logger.error(f"Database error in store-commodity: {e}")
        return jsonify({"success": False, "message": "Database error"}), 500
    except Exception as e:
        logger.error(f"Error in store-commodity: {str(e)}")
        return jsonify({"success": False, "message": "Internal server error"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/get-farmers', methods=['GET'])
def get_farmers():
    conn = None
    try:
        commodity_query = request.args.get('commodity', '').lower()
        location_query = request.args.get('location', '').lower()
        
        conn = sqlite3.connect(FARMER_DB)
        cursor = conn.cursor()
        
        query = "SELECT * FROM farmers WHERE 1=1"
        params = []
        
        if commodity_query:
            query += " AND LOWER(commodity) LIKE ?"
            params.append(f'%{commodity_query}%')
            
        if location_query:
            query += " AND LOWER(address) LIKE ?"
            params.append(f'%{location_query}%')
        
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        farmers = []
        
        for row in cursor.fetchall():
            image_url = None
            if row[8]:  # image_path field
                filename = os.path.basename(row[8])
                image_url = f"/uploads/{filename}"
                
            farmers.append({
                "id": row[0],
                "name": row[1],
                "commodity": row[2],
                "quantity": row[3],
                "price": row[4],
                "address": row[5],
                "mobile": row[6],
                "email": row[7],
                "image_url": image_url,
                "created_at": row[9]
            })
            
        return jsonify(farmers)

    except Exception as e:
        logger.error(f"Error in get-farmers: {str(e)}")
        return jsonify({"success": False, "message": "Internal server error"}), 500
    finally:
        if conn:
            conn.close()

# ====================== VENDOR ROUTES ======================
@app.route('/store-vendor', methods=['POST'])
def store_vendor():
    conn = None
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data received"}), 400

        email = data.get("email", "").strip().lower()
        if not email:
            return jsonify({"success": False, "message": "Valid email is required"}), 400

        stored_data = otp_storage.get(email)
        if not stored_data or not stored_data.get('verified'):
            return jsonify({"success": False, "message": "Email not verified"}), 403

        fields = {
            "name": data.get("name", "").strip(),
            "market_name": data.get("market_name", "").strip(),
            "commodity": data.get("commodity", "").strip(),
            "quantity": data.get("quantity"),
            "price": data.get("price"),
            "email": email
        }

        if not all(fields.values()):
            return jsonify({"success": False, "message": "All fields are required"}), 400

        try:
            fields["quantity"] = int(fields["quantity"])
            fields["price"] = float(fields["price"])
            if fields["quantity"] <= 0 or fields["price"] <= 0:
                raise ValueError("Values must be positive")
        except (ValueError, TypeError) as e:
            return jsonify({"success": False, "message": "Invalid quantity or price"}), 400

        conn = sqlite3.connect(VENDOR_DB)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO vendors (name, market_name, commodity, quantity, price, email) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            fields["name"], fields["market_name"], fields["commodity"],
            fields["quantity"], fields["price"], fields["email"]
        ))
        conn.commit()
        
        del otp_storage[email]
        
        return jsonify({
            "success": True,
            "message": "Vendor data stored successfully",
            "id": cursor.lastrowid
        })

    except sqlite3.Error as e:
        logger.error(f"Database error in store-vendor: {e}")
        return jsonify({"success": False, "message": "Database error"}), 500
    except Exception as e:
        logger.error(f"Error in store-vendor: {str(e)}")
        return jsonify({"success": False, "message": "Internal server error"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/get-vendors', methods=['GET'])
def get_vendors():
    conn = None
    try:
        commodity_query = request.args.get('commodity', '').lower()
        market_query = request.args.get('market', '').lower()
        
        conn = sqlite3.connect(VENDOR_DB)
        cursor = conn.cursor()
        
        query = "SELECT * FROM vendors WHERE 1=1"
        params = []
        
        if commodity_query:
            query += " AND LOWER(commodity) LIKE ?"
            params.append(f'%{commodity_query}%')
            
        if market_query:
            query += " AND LOWER(market_name) LIKE ?"
            params.append(f'%{market_query}%')
        
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        vendors = []
        
        for row in cursor.fetchall():
            vendors.append({
                "id": row[0],
                "name": row[1],
                "market_name": row[2],
                "commodity": row[3],
                "quantity": row[4],
                "price": row[5],
                "email": row[6],
                "created_at": row[7]
            })
            
        return jsonify(vendors)

    except Exception as e:
        logger.error(f"Error in get-vendors: {str(e)}")
        return jsonify({"success": False, "message": "Internal server error"}), 500
    finally:
        if conn:
            conn.close()

# ====================== PREDICTION ROUTES ======================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        commodity = data.get('commodity', '').strip()
        market = data.get('market', '').strip()
        
        if not commodity or not market:
            return jsonify({'error': 'Both commodity and market are required'}), 400

        logger.info(f"Processing prediction request for '{commodity}' in '{market}'")

        # Load and filter data
        df = pd.DataFrame()
        for file in DATA_FILES:
            file_path = os.path.join(os.path.dirname(__file__), file)
            if not os.path.exists(file_path):
                logger.warning(f"Data file {file_path} not found")
                continue
                
            try:
                chunk = pd.read_csv(file_path)
                filtered = chunk[
                    (chunk['Commodity name'].str.lower() == commodity.lower()) & 
                    (chunk['Market name'].str.lower() == market.lower())
                ]
                df = pd.concat([df, filtered])
            except Exception as e:
                logger.error(f"Error reading {file_path}: {str(e)}")
                traceback.print_exc()

        if df.empty:
            return jsonify({'error': f'No data found for {commodity} in {market}'}), 404

        # Preprocess data
        df = df[['Modal price for the commodity', 'Calendar Day']].copy()
        df['Calendar Day'] = pd.to_datetime(df['Calendar Day'])
        df = df.sort_values('Calendar Day').set_index('Calendar Day')

        # Load model and scaler
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        scaler_path = os.path.join(os.path.dirname(__file__), SCALER_PATH)
        
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model not found at {model_path}'}), 500
        if not os.path.exists(scaler_path):
            return jsonify({'error': f'Scaler not found at {scaler_path}'}), 500

        scaler = joblib.load(scaler_path)
        model = load_model(model_path)

        # Prepare data for prediction
        scaled_data = scaler.transform(df)
        last_input = scaled_data[-TIME_STEP:]

        # Generate predictions
        predictions = []
        for _ in range(7):
            x = last_input.reshape(1, TIME_STEP, 1)
            pred = model.predict(x, verbose=0)
            price = float(scaler.inverse_transform(pred)[0][0])
            predictions.append(round(price, 2))
            last_input = np.append(last_input[1:], pred)

        return jsonify({
            'commodity': commodity,
            'market': market,
            'predictions': predictions
        })

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ====================== OTHER ROUTES ======================
@app.route('/request-delete-otp', methods=['POST'])
def request_delete_otp():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data received"}), 400

        email = data.get("email", "").strip().lower()
        user_id = data.get("id")
        user_type = data.get("type", "").lower()

        if not all([email, user_id, user_type]):
            return jsonify({"success": False, "message": "Email, ID and type are required"}), 400

        if user_type not in ['farmer', 'vendor']:
            return jsonify({"success": False, "message": "Invalid user type"}), 400

        clean_expired_otps()

        db_file = FARMER_DB if user_type == 'farmer' else VENDOR_DB
        table = 'farmers' if user_type == 'farmer' else 'vendors'

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(f"SELECT email FROM {table} WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            return jsonify({"success": False, "message": "Record not found"}), 404
            
        db_email = result[0].strip().lower()
        if db_email != email:
            return jsonify({"success": False, "message": "Email does not match the record"}), 403

        otp = str(random.randint(100000, 999999))
        expiry = datetime.now() + timedelta(minutes=OTP_EXPIRY_MINUTES)
        
        otp_key = f"delete_{email}_{user_id}_{user_type}"
        otp_storage[otp_key] = {
            "otp": otp,
            "expiry": expiry,
            "email": email,
            "id": user_id,
            "type": user_type
        }

        if send_email(email, otp):
            return jsonify({
                "success": True,
                "message": f"OTP sent successfully. Valid for {OTP_EXPIRY_MINUTES} minutes."
            })
        return jsonify({"success": False, "message": "Failed to send OTP"}), 500

    except Exception as e:
        logger.error(f"Error in request-delete-otp: {str(e)}")
        return jsonify({"success": False, "message": "Internal server error"}), 500

@app.route('/verify-and-delete', methods=['POST'])
def verify_and_delete():
    conn = None
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data received"}), 400

        user_id = data.get("id")
        email = data.get("email", "").strip().lower()
        otp = data.get("otp", "").strip()
        user_type = data.get("type", "").lower()

        if not all([user_id, email, otp, user_type]):
            return jsonify({"success": False, "message": "ID, email, OTP and type are required"}), 400

        if user_type not in ['farmer', 'vendor']:
            return jsonify({"success": False, "message": "Invalid user type"}), 400

        clean_expired_otps()

        otp_key = f"delete_{email}_{user_id}_{user_type}"
        stored_data = otp_storage.get(otp_key)
        
        if not stored_data:
            return jsonify({"success": False, "message": "OTP expired or not found"}), 404

        if stored_data["otp"] != otp:
            return jsonify({"success": False, "message": "Invalid OTP"}), 400

        db_file = FARMER_DB if user_type == 'farmer' else VENDOR_DB
        table = 'farmers' if user_type == 'farmer' else 'vendors'

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        image_path = None
        if user_type == 'farmer':
            cursor.execute(f"SELECT image_path FROM {table} WHERE id = ? AND email = ?", (user_id, email))
            result = cursor.fetchone()
            if result:
                image_path = result[0]

        cursor.execute(f"DELETE FROM {table} WHERE id = ?", (user_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            return jsonify({"success": False, "message": "No record deleted"}), 404

        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                logger.error(f"Error deleting image file: {str(e)}")

        del otp_storage[otp_key]
        logger.info(f"Successfully deleted record {user_id} from {table}")

        return jsonify({
            "success": True,
            "message": "Entry deleted successfully"
        })

    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({"success": False, "message": "Database error"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in verify-and-delete: {str(e)}")
        return jsonify({"success": False, "message": "Internal server error"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create required directories
    ensure_upload_folder()
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)