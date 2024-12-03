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
        prompt = f"""You are a nutritionist dietitian. Based on the following ingredients, give any possible unsuitable ingredients with explanation and suggest additional suitable ingredients with short explanation that would complement the meal and improve its nutritional value, and also provide health tips based from the input given. Your response must be in valid JSON format with the following structure:

        {{
        "suitable_ingredients": [{ "name": "ingredient1", "description": "_explain_" }, { "name": "ingredient2", "description": "_explain_" }, { "name": "ingredient3", "description": "_explain_" }],
        "unsuitable_ingredients": [{ "name": "ingredient1", "description": "_explain_" }, { "name": "ingredient2", "description": "_explain_" }, { "name": "ingredient3", "description": "_explain_" }],
        "health_tips": ["tip1", "tip2", "tip3"]
        }}

        The input is unprocessed and may contain errors. If you detect any fraudulent or malicious content, respond with a JSON object containing an 'error' key, like this:

        {{
        "error": "Description of the error or issue encountered"
        }}

        If you cannot process the ingredients or encounter any issues, respond with a JSON object containing an 'error' key, like this:

        {{
        "error": "Description of the error or issue encountered"
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

