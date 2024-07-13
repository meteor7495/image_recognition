from flask import Flask, render_template, request, jsonify
from models.combined_model import CombinedModel

app = Flask(__name__)

# Initialize your combined model
combined_model = CombinedModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Perform prediction
    result = combined_model.predict(file)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
