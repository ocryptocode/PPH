# main.py
from openai_integration import OpenAIIntegration
from pipeline import DataAnalysisPipeline
import pandas as pd

# Initialize OpenAIIntegration and DataAnalysisPipeline
openai_api_key = 'your-openai-api-key'
openai_integration = OpenAIIntegration(openai_api_key)
pipeline = DataAnalysisPipeline(openai_integration)

# Example DataFrame
df = pd.DataFrame({'text': ['Example text 1', 'Example text 2', 'Example text 3']})

# Process DataFrame using the pipeline
pipeline.process_dataframe(df)
