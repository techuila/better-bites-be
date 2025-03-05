from flask import Flask, jsonify, request
from libs.model import Nutritionist
from libs.utils import handle_json_output, validate_json_input
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Debugging: Check if API key is loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Error: GROQ_API_KEY is not set. Check your .env file.")

# Initialize Nutritionist (ensure API key is loaded)
nutritionist = Nutritionist()

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
@handle_json_output(json_error_message=nutritionist.model_response_error)
@validate_json_input(keys=['ingredients', 'user_profile'])  # Updated to include user_profile
def analyze_ingredients():
    data = request.get_json()
    ingredients = data.get('ingredients', [])  # Default to empty list if not provided
    user_profile = data.get('user_profile', '')  # Default to empty string if not provided

    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400

    # Pass both ingredients and user_profile to the updated Nutritionist method
    return nutritionist.get_advice_from_ingredients(ingredients, user_profile)

if __name__ == '__main__':
    app.run(port=3000, debug=True)