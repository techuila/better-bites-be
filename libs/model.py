from flask import Flask, jsonify, request
from groq import Groq
from dotenv import load_dotenv
import os
import json
import logging
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_api_key():
    """Fetch API key from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.error("Error: GROQ_API_KEY is not set. Check your .env file.")
        raise ValueError("Error: GROQ_API_KEY is not set. Check your .env file.")
    return api_key

class Nutritionist:
    """
    Nutritionist class to analyze food ingredients for education and health awareness.
    - Uses the Groq API to generate analysis.
    - Bases analysis on user profile: age, sex, height, weight, and health conditions.
    """
    MODEL = "llama3-70b-8192"
    model_response_error = "Model returned an invalid JSON response."

    def __init__(self):
        """Initialize Groq API client."""
        self.client = Groq(api_key=get_api_key())

    def extract_amounts(self, ingredient):
        """Extract %, g, or ml values from an ingredient name."""
        match = re.search(r"(\d+(\.\d+)?\s*(%|g|ml))", ingredient)
        return match.group(0) if match else None

    def get_advice_from_ingredients(self, ingredients, user_profile):
        """
        Analyze ingredients and return educational nutritional advice.
        - Uses age, sex, height, weight, and health conditions for context.
        - Highlights nutritional benefits, health risks (e.g., allergies), and allergens.
        """
        # Ensure all ingredients are strings
        if not all(isinstance(ing, str) for ing in ingredients):
            return jsonify({"error": "Invalid input. Ingredients must be a list of strings."}), 400

        # Identify ingredients with amounts (%, g, ml)
        ingredients_with_amounts = {
            ing: self.extract_amounts(ing) for ing in ingredients if self.extract_amounts(ing)
        }

        # Prompt using complete user profile for education and health awareness
        prompt = f"""
        You are a professional nutritionist. Analyze the given ingredients with amounts (e.g., %, g, ml) and the user’s health profile to educate about nutrition and health awareness. Use the user’s age, sex, height, weight, and health conditions (if any) to provide context, such as general nutritional needs or health risks.

        1. **List suitable ingredients** with their nutritional benefits and general recommended intake, adjusted for the user’s profile (e.g., age, sex, weight considerations).
        2. **Identify unsuitable ingredients** based on health conditions (e.g., allergies) if provided, or general health concerns if not, with alternatives to broaden understanding.
        3. **Categorize all allergens separately**, including common allergens (e.g., nuts, shellfish, dairy) regardless of user profile, explaining why they’re allergens and potential reactions.
        4. **Ensure all provided ingredients are accounted for** in the response (suitable, unsuitable, or allergens).
        5. **For ingredients with amounts (e.g., %, g, ml)**, explain their nutritional impact based on the user’s profile (e.g., calorie density relative to weight, sodium for blood pressure).
        6. **In the Health Tips section**, provide general educational advice about all scanned ingredients, tailored to the user’s profile where applicable (e.g., allergy awareness, weight management).

        Ingredients with amounts: {ingredients_with_amounts}

        User Profile:
        {user_profile}

        Ingredients: {ingredients}

        **Response Format (ONLY return valid JSON)**:
        {{
            "title": "Ingredient Analysis",
            "suitable_ingredients": [
                {{
                    "name": "Ingredient Name (if applicable, include volume: e.g., 30g, 250ml, 10%)",
                    "description": "Provide a clear explanation of the ingredient's nutritional benefits and why it’s generally healthy, considering age, sex, height, and weight.",
                    "Recommended Intake": "Specify a general recommended intake amount, adjusted for the user’s profile (e.g., 30g per meal, consume in moderation)."
                }}
            ],
            "unsuitable_ingredients": [
                {{
                    "name": "Ingredient Name (if applicable, include volume: e.g., 30g, 250ml, 10%)",
                    "description": "Provide a clear explanation of why this ingredient might be a concern, based on health conditions or general nutrition relative to the user’s profile.",
                    "alternatives": "Suggest healthier or safer options to explore (e.g., 10g of a substitute)."
                }}
            ],
            "allergens": [
                {{
                    "name": "Ingredient Name (if applicable, include volume: e.g., 30g, 250ml, 10%)",
                    "description": "Provide a clear explanation of why this ingredient is considered a common allergen.",
                    "Potential Reaction": "Explain how it may affect sensitive individuals (e.g., allergic reactions, digestive issues)."
                }}
            ],
            "health_tips": [
                {{
                    "name": "Educational Health Tip",
                    "description": "Provide a clear explanation of a health concept related to these ingredients, tailored to the user’s age, sex, height, weight, and health conditions.",
                    "suggestion": "Give actionable advice for better health awareness (e.g., monitor portions, avoid allergens)."
                }}
            ]
        }}

        If the ingredients are invalid or unrelated to food, return:
        {{
            "error": "Invalid input. Provide valid food ingredients."
        }}

        JSON Response:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": "You are a nutritionist dietitian. Always respond in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False
            )

            response_text = response.choices[0].message.content.strip()
            logging.info(f"Model Response: {response_text}")

            # Validate JSON format
            try:
                parsed_response = json.loads(response_text)
                return jsonify(parsed_response)
            except json.JSONDecodeError:
                logging.error("Invalid JSON received from model.")
                return jsonify({"error": self.model_response_error}), 500

        except Exception as e:
            logging.error(f"Error encountered: {e}")
            return jsonify({"error": "Error encountered from Groq API"}), 500

