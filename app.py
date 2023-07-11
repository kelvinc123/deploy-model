from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)

    # Convert data into numpy array
    predict_request = np.array(data['image'])

    # Normalizing data
    predict_request = predict_request / 255.0

    # Use the model to predict
    prediction = model.predict(np.expand_dims(predict_request, 0))

    # Get the first and only result
    output = prediction[0].argmax()

    # Return the result as JSON
    return jsonify({
        "Prediction": str(output)
    }), 200

if __name__ == '__main__':
    model = tf.keras.models.load_model('model1.h5')
    app.run(host='0.0.0.0', port=5001, debug=True)