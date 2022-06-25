from flask import Flask, jsonify, request
from Classify import getPred

app = Flask(__name__)

@app.route('/predict-letter', methods = ['POST'])

def readDigit():
    image = request.files.get('letter')
    prediction = getPred(image)
    return jsonify({
        'prediction' : prediction
    }), 200

if __name__ == '__main__':
    app.run(debug = True)