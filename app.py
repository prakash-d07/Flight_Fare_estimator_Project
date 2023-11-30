from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
from src.Flight_Fare_estimator_Project import logger

app = Flask(__name__)
CORS(app)


# Load the StandardScaler object from the pickle file
scaler_filename = 'artifacts/data_modelling/scaler.pkl'
with open(scaler_filename, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the model from the file (replace 'model.pkl' with your actual model file)
model_filename = 'artifacts/data_modelling/xgb_model.pkl'
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html')

def preprocess_input(input_data):
    # Mapping for Airlines
    airlines = [
        'Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers',
        'SpiceJet', 'Vistara', 'Air Asia','GoAir', 'Multiple carriers Premium economy',
        'Jet Airways Business', 'Vistara Premium economy', 'Trujet'
    ]
    airline_list = [1 if input_data['Airline'] == airline else 0 for airline in airlines]

    # Mapping for Sources
    sources = ['Delhi', 'Kolkata', 'Banglore', 'Mumbai', 'Chennai']
    source_list = [1 if input_data['Source'] == source else 0 for source in sources]

    # Mapping for Destinations
    destinations = ['Cochin', 'Banglore', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata']
    destination_list = [1 if input_data['Destination'] == destination else 0 for destination in destinations]

    # Mapping for Total_Stops
    stops_list = [1 if input_data['Total_Stops'] == stop else 0 for stop in ['1 stop', '2 stops', '3 stops', '4 stops', 'non-stop']]

    numerical_values = [
        input_data ['Month_of_Month_of_journey'],
        input_data['Day_of_Date_of_Journey'],
        input_data['Duration_minutes']
    ]

    # Standardize numerical values
    scaled_numerical_values = scaler.transform([numerical_values]).reshape(1, -1)

    # Combine the scaled numerical values with the categorical values
    scaled_input_data = np.concatenate([scaled_numerical_values[0], airline_list, source_list, destination_list, stops_list])

    # Reshape for prediction
    final_input_data = scaled_input_data.reshape(1, -1)

    logger.info(f"scaled input data for prediction is {scaled_input_data}")
    logger.info(f"scaled input data for prediction is {final_input_data}")

    return final_input_data

# Use the same function for both routes
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        # Log the request data
        logger.info("Request Data:", data)

        # Preprocess the input data
        input_data = preprocess_input(data)

        # Make predictions with the loaded model
        prediction = loaded_model.predict(input_data)

        output = round(prediction[0], 2)

        return jsonify({"prediction_text": "Your Flight price is Rs. {}".format(output)})

    except Exception as e:
        # Log the error
        logger.error("Error:", e)
        return jsonify({"prediction_text": "Error: {}".format(e)})
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Accesing Data from the form")
        # Extracting data from the form
        input_data = {
            'Airline': request.form['Airline'],
            'Source': request.form['Source'],
            'Destination': request.form['Destination'],
            'Total_Stops': request.form['Total_Stops'],
            'Month_of_Month_of_journey': float(request.form['Month_of_Month_of_journey']),
            'Day_of_Date_of_Journey': float(request.form['Day_of_Date_of_Journey']),
            'Duration_minutes': float(request.form['Duration_minutes'])
        }

        # Preprocess the input data
        final_input_data = preprocess_input(input_data)

        # Assuming 'loaded_model' is your model object
        prediction = loaded_model.predict(final_input_data)

        # Assuming 'logger' is your logger object
        logger.info("Prediction:", prediction)

        # Assuming your output is numeric, you can round it to 2 decimal places
        output = round(prediction[0], 2)

        return render_template("home.html", prediction_text="Your prediction is: {}".format(output))

    except Exception as e:
        # Log the error
        logger.error("Error:", e)
        return render_template("home.html", prediction_text="Error: {}".format(e))

if __name__ == '__main__':
    app.run(debug=True)








