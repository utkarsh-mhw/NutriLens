from flask import Flask, jsonify, request
from flask_cors import CORS
from src.analyze_product import * 
import json

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

with open('frontend/public/demo_snapshots/precomputed_payloads.json', 'r') as f:
    PRECOMPUTED_PAYLOADS = json.load(f)

@app.route('/api/analyze_precomputed', methods=['POST'])
def analyze_precomputed():
    data = request.json
    product_name = data.get('product_name')
    
    if product_name not in PRECOMPUTED_PAYLOADS:
        return jsonify({
            'success': False,
            'error': f'Product "{product_name}" not found in results, please contact developer'
        }), 404
    
    # Return the precomputed payload directly
    return jsonify(PRECOMPUTED_PAYLOADS[product_name])

if __name__ == '__main__':
    app.run(debug=True, port=5002)

