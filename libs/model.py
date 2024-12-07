from flask import jsonify
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

class Nutritionist:
    MODEL = "llama3-70b-8192"
    model_response_error = "Model returned an invalid json response."

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        pass

    def get_advice_from_ingredients(self, ingredients):
        prompt = f"""You are a nutritionist dietitian. Based on the following ingredients, give a title, any possible unsuitable ingredients (unsuitable_ingredients) with explanation and suggest suitable ingredients (suitable_ingredients) with short explanation that would complement the meal and improve its nutritional value, and also provide health tips (health_tips) with title and description based from the input given. Your response must be ONLY in valid JSON format with the following structure:

        {{
        "suitable_ingredients": [{{ "name": "ingredient1", "description": "_explain_" }}, {{ "name": "ingredient2", "description": "_explain_" }}, {{ "name": "ingredient3", "description": "_explain_" }}],
        "unsuitable_ingredients": [{{ "name": "ingredient1", "description": "_explain_" }}, {{ "name": "ingredient2", "description": "_explain_" }}, {{ "name": "ingredient3", "description": "_explain_" }}],
        "health_tips": [{{ "name": "tip1", "description": "_explain_" }}, {{ "name": "tip2", "description": "_explain_" }}, {{ "name": "tip3", "description": "_explain_" }}],
        "title": "Title of the meal"
        }}

        The input is unprocessed and may contain errors. If you detect any fraudulent, malicious content, or you cannot process the ingredients, respond only a JSON object containing an 'error' key with a value "Invalid input. The provided ingredients are not edible or related to food. Please provide a list of valid ingredients to process." like this:

        {{
        "error": "Invalid input. The provided ingredients are not edible or related to food. Please provide a list of valid ingredients to process."
        }}

        Ensure your response can be parsed by Python's json.loads() function.

        Ingredients: {ingredients}

        JSON Response:"""

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": "You are a nutritionist dietitian. Always respond in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            print(response.choices[0].message.content)

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return jsonify({"error": "Error encountered from Groq API"}), 500

