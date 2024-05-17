# openai_integration.py
import openai

class OpenAIIntegration:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key

    def generate_text_with_gpt3(self, prompt):
        # Call OpenAI's GPT-3 model to generate text based on prompt
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()
