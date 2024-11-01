from flask import Flask, request, jsonify
import psycopg2
import numpy as np
from PIL import Image
import io
from tensorflow import keras
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import logging
from logging.handlers import RotatingFileHandler
import flask_monitoringdashboard as dashboard
from apscheduler.schedulers.background import BackgroundScheduler

# Create Flask app and API instance
app = Flask(__name__)
api = Api(app, version='1.0', title='ML Model API',
          description='API for image classification and customer purchase frequency prediction')

ns = api.namespace('predictions', description='Prediction operations')

# Set up logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Add monitoring dashboard
dashboard.bind(app)

# PostgreSQL connection function
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="ml_course",
        user="postgres",
        password="admin"
    )

# Load models
img_model = keras.models.load_model('image_classification_model.h5')
cust_model = keras.models.load_model('customer_purchase_model.h5')

# Define expected input data structures
image_upload = api.parser()
image_upload.add_argument('file', location='files', type=FileStorage, required=True)

customer_input = api.model('Customer', {
    'age': fields.Integer(required=True, description='Customer age'),
    'income': fields.Float(required=True, description='Customer income')
})

@ns.route('/classify_image')
class ImageClassification(Resource):
    @ns.expect(image_upload)
    @ns.response(200, 'Success')
    @ns.response(400, 'Validation Error')
    def post(self):
        '''Classify an uploaded image'''
        try:
            args = image_upload.parse_args()
            file = args['file']
            
            if file.filename == '':
                return {'error': 'No file selected'}, 400
            
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            prediction = img_model.predict(img_array)
            predicted_class = int(prediction.round())
            
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO synthetic_images (image, label)
                        VALUES (%s, %s) RETURNING id
                    """, (psycopg2.Binary(img_bytes), predicted_class))
                    image_id = cur.fetchone()[0]
                    conn.commit()
            
            return {'image_id': image_id, 'predicted_class': predicted_class}
        except Exception as e:
            app.logger.error(f'Error in classify_image: {str(e)}')
            return {'error': str(e)}, 500

@ns.route('/predict_purchase_frequency')
class PurchaseFrequencyPrediction(Resource):
    @ns.expect(customer_input)
    @ns.response(200, 'Success')
    @ns.response(400, 'Validation Error')
    def post(self):
        '''Predict customer purchase frequency'''
        try:
            data = api.payload
            age = data['age']
            income = data['income']
            
            if age < 0 or income < 0:
                return {'error': 'Age and income must be non-negative'}, 400
            
            input_data = np.array([[age, income]])
            prediction = cust_model.predict(input_data)
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            class_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
            predicted_label = class_labels[predicted_class]
            
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO synthetic_customers (age, income, purchase_frequency)
                        VALUES (%s, %s, %s) RETURNING id
                    """, (age, income, predicted_label))
                    customer_id = cur.fetchone()[0]
                    conn.commit()
            
            return {'customer_id': customer_id, 'predicted_purchase_frequency': predicted_label}
        except Exception as e:
            app.logger.error(f'Error in predict_purchase_frequency: {str(e)}')
            return {'error': str(e)}, 500

# Model retraining function
def retrain_models():
    try:
        conn = get_db_connection()
        
        # Retrain image classification model
        cur = conn.cursor()
        cur.execute("SELECT image, label FROM synthetic_images")
        image_data = cur.fetchall()
        
        X_images = np.array([np.frombuffer(row[0], dtype=np.uint8).reshape((28, 28)) for row in image_data])
        y_images = np.array([row[1] for row in image_data])
        
        img_model.fit(X_images / 255.0, y_images, epochs=5, validation_split=0.2)
        img_model.save('image_classification_model.h5')
        
        # Retrain customer prediction model
        cur.execute("SELECT age, income, purchase_frequency FROM synthetic_customers")
        customer_data = cur.fetchall()
        
        X_customers = np.array([[row[0], row[1]] for row in customer_data])
        y_customers = np.array([row[2] for row in customer_data])
        
        cust_model.fit(X_customers.astype(np.float32), pd.get_dummies(y_customers).values,
                       epochs=10, validation_split=0.2)
        cust_model.save('customer_purchase_model.h5')
        
        app.logger.info('Models retrained successfully')
    except Exception as e:
        app.logger.error(f'Error during model retraining: {str(e)}')

# Schedule model retraining every week using BackgroundScheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=retrain_models, trigger="interval", weeks=1)
scheduler.start()

if __name__ == '__main__':
    app.run(debug=True)
