# pipeline.py
from model_functions import analyze_text_sentiment_with_bert
from openai_integration import OpenAIIntegration

class DataAnalysisPipeline:
    def __init__(self, openai_api_key):
        self.openai_integration = OpenAIIntegration(openai_api_key)

    def process_dataframe(self, df):
        for index, row in df.iterrows():
            text = row['text']
            sentiment = analyze_text_sentiment_with_bert(text)
            generated_text = self.openai_integration.generate_text_with_gpt3(text)
            # Process sentiment and generated text as needed
            # Example: store results in a new DataFrame or database
