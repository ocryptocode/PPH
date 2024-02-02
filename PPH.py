# Step 1: Creating my application
from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

app = Flask(__name__)


@app.route('/hello')
def hello_world():
    return "hello world"


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

# Step 2: Disposing the database
# Read XLS file using pandas library into a Pandas DataFrame
xls_file_path = r'C:\Projet Oussama\Base de données.xlsx'
df = pd.read_excel(xls_file_path)

# Display the database file
print(df)

# Add new rows to the database table and a function to it
new_rows = [
    {"nombre": 1, "nom": "Dupont", "prénom": "Jean", "nom du père": "Dupont père", "nbre de tel du père": 123456789,
     "nom de la mère": "Dupont mère", "nbre de tel de la mère": 987654321, "date de naissance": "01-01-2005",
     "classe": "10A", "Age": 17, "Mode de paiement": "Carte bancaire", "Statut de paiement": "validé",
     "date du paiement": "05-02-2023"},
    {"nombre": 2, "nom": "Martin", "prénom": "Marie", "nom du père": "Martin père", "nbre de tel du père": 987654321,
     "nom de la mère": "Martin mère", "nbre de tel de la mère": 123456789, "date de naissance": "15-08-2003",
     "classe": "11B", "Age": 14, "Mode de paiement": "Chèque", "Statut de paiement": "non validé",
     "date du paiement": "20-09-2022"},
    {"nombre": 3, "nom": "Dubois", "prénom": "Ahmed", "nom du père": "Dubois père", "nbre de tel du père": 1122334455,
     "nom de la mère": "Dubois mère", "nbre de tel de la mère": 5544332211, "date de naissance": "03-04-2006",
     "classe": "9C", "Age": 16, "Mode de paiement": "Virement", "Statut de paiement": "non validé",
     "date du paiement": "10-03-2023"},
    {"nombre": 4, "nom": "Leroux", "prénom": "Sophie", "nom du père": "Leroux père", "nbre de tel du père": 9988776655,
     "nom de la mère": "Leroux mère", "nbre de tel de la mère": 5543537281, "date de naissance": "02-05-2004",
     "classe": "10A", "Age": 18, "Mode de paiement": "Carte bancaire", "Statut de paiement": "validé",
     "date du paiement": "15-01-2023"},
    {"nombre": 5, "nom": "Garcia", "prénom": "Carlos", "nom du père": "Garcia père", "nbre de tel du père": 3344556677,
     "nom de la mère": "Garcia mère", "nbre de tel de la mère": 8899001122, "date de naissance": "07-06-2002",
     "classe": "12B", "Age": 20, "Mode de paiement": "Virement", "Statut de paiement": "validé",
     "date du paiement": "02-04-2023"},
    {"nombre": 6, "nom": "Chen", "prénom": "Mei", "nom du père": "Chen père", "nbre de tel du père": 5566778899,
     "nom de la mère": "Chen mère", "nbre de tel de la mère": 9988776655, "date de naissance": "12-09-2005",
     "classe": "10A", "Age": 16, "Mode de paiement": "Chèque", "Statut de paiement": "non validé",
     "date du paiement": "18-12-2022"},
    {"nombre": 7, "nom": "Ahmed", "prénom": "Fatima", "nom du père": "Ahmed père", "nbre de tel du père": 1122334455,
     "nom de la mère": "Ahmed mère", "nbre de tel de la mère": 3344556677, "date de naissance": "25-03-2003",
     "classe": "11B", "Age": 19, "Mode de paiement": "Carte bancaire", "Statut de paiement": "validé",
     "date du paiement": "08-09-2022"},
    {"nombre": 8, "nom": "Smith", "prénom": "Emily", "nom du père": "Smith père", "nbre de tel du père": 8899001122,
     "nom de la mère": "Smith mère", "nbre de tel de la mère": 5566778899, "date de naissance": "14-08-2004",
     "classe": "10A", "Age": 18, "Mode de paiement": "Virement", "Statut de paiement": "non validé",
     "date du paiement": "05-05-2023"},
    {"nombre": 9, "nom": "Wang", "prénom": "Jun", "nom du père": "Wang père", "nbre de tel du père": 9988776655,
     "nom de la mère": "Wang mère", "nbre de tel de la mère": 8899001122, "date de naissance": "19-06-2002",
     "classe": "12B", "Age": 20, "Mode de paiement": "Chèque", "Statut de paiement": "non validé",
     "date du paiement": "14-03-2023"},
    {"nombre": 10, "nom": "Park", "prénom": "Min", "nom du père": "Park père", "nbre de tel du père": 3344556677,
     "nom de la mère": "Park mère", "nbre de tel de la mère": 5566778899, "date de naissance": "22-11-2003",
     "classe": "11B", "Age": 19, "Mode de paiement": "Carte bancaire", "Statut de paiement": "validé",
     "date du paiement": "28-02-2022"}
]

df = pd.concat([df, pd.DataFrame(new_rows)])

# Display the updated database file
print(df)


# Function to check if a payment is validated or not
def check_payment_status(row):
    if row["Statut de paiement"] == "validé":
        return "a jour avec le paiement"
    else:
        return


# Apply the function to each row in the DataFrame and create a new column
df["Validation du paiement"] = df.apply(check_payment_status, axis=1)

# Display the updated database file with the new column
print(df)

# Step 3: Implementing a text classification model
# Load the dataset
dataset = pd.read_csv('text_classification_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['label'], test_size=0.2, random_state=42)

# Create a text classification pipeline using TF-IDF and Logistic Regression
text_classification_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the model
text_classification_pipeline.fit(X_train, y_train)

# Make predictions on the test set
predictions = text_classification_pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Text Classification Model Accuracy: {accuracy}')


# Step 4 : implementing the sequence to sequence model and the custom transformer
class CustomTransformer (nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(CustomTransformer, self).__init__()

        # Self-attention encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers
        )

        # Linear layer for final output
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Forward pass through transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Linear layer for final output
        x = self.linear(x)
        return x


class Seq2SeqWithCustomTransformer (nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(Seq2SeqWithCustomTransformer, self).__init__()

        # Custom transformer encoder
        self.custom_transformer_encoder = CustomTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            output_size=output_size
        )

        # LSTM-based decoder (you can replace this with a transformer decoder if needed)
        self.decoder = nn.LSTM(input_size=output_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input_seq):
        # Forward pass through custom transformer encoder
        encoder_output = self.custom_transformer_encoder(input_seq)

        # You can modify the decoder input based on your specific task
        # For simplicity, we use the encoder output as the initial hidden state for the decoder
        decoder_output, _ = self.decoder(encoder_output.unsqueeze(1))

        return decoder_output


# Step 5: Using BERT for sequence classification
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize input text and convert to PyTorch tensors
input_text = "This is a sample text for BERT classification."
tokenized_input = tokenizer(input_text, return_tensors='pt')
outputs = bert_model(**tokenized_input)

# Step 6 : Using GPT-2 for text generation
# Load pre-trained GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text using GPT-2
input_prompt = "Once upon a time in a"
input_ids = gpt2_tokenizer.encode(input_prompt, return_tensors='pt')
output_text = gpt2_model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2,
                                  top_k=50, top_p=0.95, temperature=0.7)

generated_text = gpt2_tokenizer.decode(output_text[0], skip_special_tokens=True)
print(f'Generated Text: {generated_text}')


# Step 7 : Implementing a custom class and using it in a pipeline
class MyClass:
    def __init__(self, database):
        self.database_mapping = database

    def transform(self):
        """
        This method transforms input data and updates the class attributes.
        """
        self.database_mapping['number'] = input("Enter a number: ")
        self.database_mapping['name'] = input("Enter a name: ")
        self.database_mapping['surname'] = input("Enter a surname: ")
        self.database_mapping['father name'] = input("Enter father's name: ")
        self.database_mapping['father phone'] = input("Enter father's phone number: ")
        self.database_mapping['mother name'] = input("Enter mother's name: ")
        self.database_mapping['mother phone'] = input("Enter mother's phone number: ")
        self.database_mapping['date of birth'] = input("Enter date of birth: ")
        self.database_mapping['class'] = input("Enter class: ")
        self.database_mapping['age'] = input("Enter age: ")
        self.database_mapping['payment method'] = input("Enter payment method: ")
        self.database_mapping['payment status'] = input("Enter payment status: ")
        self.database_mapping['payment update'] = input("Is someone's payment updated? (yes/no): ").lower()
        self.database_mapping['payment date'] = input("Enter payment date: ")

        return self.database_mapping


class TextTransformer:
    def fit(self, x, y=None):
        for step_name, step_instance in self.steps:
            step_instance.fit(x, y)
        return self

    def transform(self, x):
        transformed_data = x
        for step_name, step_instance in self.steps:
            transformed_data = step_instance.transform(transformed_data)
        return transformed_data


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit_transform(self, input_data):
        x = input_data

        for step_name, step_instance in self.steps:
            if isinstance(step_instance, MyClass):
                x = step_instance.transform()

            # Assuming the step is an instance of TextTransformer
            elif isinstance(step_instance, TextTransformer):
                x = step_instance.transform(x)
        return x


# Step 8: Model training with PyTorch
# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Dummy data for PyTorch model training
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))


# Define a simple neural network model
class SimpleNN (nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Initialize the model, loss function, and optimizer
model = SimpleNN(input_size=10, hidden_size=5, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    # Forward pass
    outputs = model
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item ()}')

# Step 9 : Using DataLoader for efficient data loading
# Dummy dataset and DataLoader
dummy_dataset = TensorDataset(X_train, y_train)
batch_size = 32
dataloader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

# Iterate over batches in DataLoader
for inputs, labels in dataloader:
    # Training logic for each batch
    pass

# Step 10 : Fine-tuning BERT model
# Load pre-trained BERT model for fine-tuning
fine_tuned_bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = AdamW(fine_tuned_bert_model.parameters(), lr=2e-5)

# Fine-tuning loop
num_epochs = 3
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = fine_tuned_bert_model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Step 11 : Implementing learning rate scheduling
# Learning rate scheduler for PyTorch optimizer
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=100)

# Training loop with learning rate scheduling
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = fine_tuned_bert_model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()


