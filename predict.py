from flask import Flask
from flask import request
from flask import jsonify
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

model_path = '/app/terrain-classification.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = ['AnnualCrop',
 'Forest',
 'HerbaceousVegetation',
 'Highway',
 'Industrial',
 'Pasture',
 'PermanentCrop',
 'Residential',
 'River',
 'SeaLake']

app = Flask('terrain')

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    img_url = data['url'] 
    print(img_url)
    preprocessor = create_preprocessor('xception', target_size=(150, 150))
    X = preprocessor.from_url(img_url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return jsonify(dict(zip(classes, float_predictions)))

@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)