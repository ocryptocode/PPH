# main.py
from flask import Flask, render_template
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
from openai_integration import OpenAIIntegration
from utils import setup_logging, log_info, log_error
from pipeline import DataAnalysisPipeline

# Initialize OpenAIIntegration and DataAnalysisPipeline
openai_api_key = 'your-openai-api-key'
openai_integration = OpenAIIntegration(openai_api_key)
pipeline = DataAnalysisPipeline(openai_integration)

# Example DataFrame
df = pd.DataFrame({'text': ['Example text 1', 'Example text 2', 'Example text 3']})

# Process DataFrame using the pipeline
pipeline.process_dataframe(df)

app = Flask(__name__)

# Initialize logging
setup_logging('app.log')

# Load OpenAI API key
openai_integration = OpenAIIntegration(api_key=os.getenv('OPENAI_API_KEY'))

# Load data
file_path = os.getenv('File_PPH_accdb', r'C:\Users\LENOVO\PycharmProjects\PPH\DF.accdb')
df = pd.read_csv(file_path)


# Function to check if a payment is validated
def check_payment_status(row):
	if row['Payment statement'] == 'validated':
		return 'updated with the paiements'
	else:
		return None


# Apply the function to each row in the DataFrame
df["Validation du paiement"] = df.apply(check_payment_status, axis=1)

# Read payments data and merge with main DataFrame
payments_file = os.getenv('File_PPH_xls', r'C:\Users\LENOVO\PycharmProjects\PPH\Paiements.xlsx')
payments_df = pd.read_excel(payments_file)
merged_data = pd.merge(df, payments_df, on='nombre', how='inner')


# Function to generate payment statements
def generate_payment_statements(df):
	from docx import Document

	document = Document()
	for index, row in df.iterrows():
		document.add_heading(f'Reçu de paiement pour {row["nombre"]}', level=1)
		document.add_paragraph(f'Montant : {row["Montant"]}')
		document.add_paragraph(f'Date du paiement : {row["Date du paiement"]}')
		# Add more details as needed
		document.add_page_break()

	document.save('recus_paiements.docx')


# Generate payment statements
generate_payment_statements(merged_data)


@app.route('/hello')
def hello_world():
	return "Hello, World!"


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyze')
def analyze():
	# Example text for BERT sentiment analysis
	input_text = "This is a sample text for BERT classification."
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
	inputs = tokenizer(input_text, return_tensors='pt')
	outputs = model(**inputs)
	sentiment = torch.argmax(outputs.logits, dim=1).item()

	# Example text generation using GPT-2
	input_prompt = "Once upon a time in a"
	gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
	input_ids = gpt2_tokenizer.encode(input_prompt, return_tensors='pt')
	output_text = gpt2_model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2,
	                                  top_k=50, top_p=0.95, temperature=0.7)
	generated_text = gpt2_tokenizer.decode(output_text[0], skip_special_tokens=True)

	log_info(f"Sentiment: {sentiment}, Generated Text: {generated_text}")
	return f"Sentiment: {sentiment}, Generated Text: {generated_text}"


if __name__ == '__main__':
	app.run(debug=True)
