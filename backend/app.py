from flask import Flask, jsonify, request
from flask_cors import CORS
from src.analyze_product import *

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    product_name = data.get('product_name', 'Tortellini rosa')
    
    if not product_name:
        return jsonify({'error': 'product_name required'}), 400
    
    result = analyze_product(product_name)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5002)