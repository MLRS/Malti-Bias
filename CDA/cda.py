!pip install transformers datasets torch

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import BertTokenizer, BertModel
# Load the CSV file
data = pd.read_csv('bertu_flores_cda.csv')
print(data)

# Select only the 'sent' column and add a dummy label column
data = data[['sent']]
data['label'] = 0.0  # Add a dummy label
# Limit the dataset to 100,000 rows
data = data.head(30000)

# Convert the DataFrame to a Dataset
dataset = Dataset.from_pandas(data)
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)  # Change num_labels based on your task
# Load model directly
tokenizer = AutoTokenizer.from_pretrained("MLRS/mBERTu")
model = AutoModelForSequenceClassification.from_pretrained("MLRS/mBERTu", num_labels=1)  # Update to use 1 label for dummy
# Define the tokenization function
# def tokenize_function(example):
#     return tokenizer(example['sent'], truncation=True)

# Apply the tokenization function to the dataset
# tokenized_dataset = dataset.map(tokenize_function, batched=True)
# # Define the tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['sent'],
        padding='max_length',  # Ensure padding to max length
        truncation=True,        # Ensure truncation to max length
        max_length=512         # Set the maximum length
    )

# # Apply the tokenization function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Ensure labels are float
def format_labels(examples):
    examples["label"] = float(examples["label"])
    return examples

tokenized_dataset = tokenized_dataset.map(format_labels)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',
    eval_steps=10,  # Set evaluation steps to match save steps if needed
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Match with command line args
    per_device_eval_batch_size=8,  # You can keep this as 8 if you want
    max_steps=10,  # Use max_steps instead of num_train_epochs
    weight_decay=0.01,
    save_steps=10,  # Ensure this matches the save_steps parameter from the script
    seed=0,
    gradient_accumulation_steps=128,  # Match with command line args
    logging_dir='./logs',  # Optional, specify the logging directory
    logging_steps=10
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-mbert')
tokenizer.save_pretrained('./fine-tuned-mbert')

#crows
!python metric.py --input_file="crowspairs_mt_final.csv" --lm_model="mbertu" --output_file="output.csv"

#seat
!sh run_seat_debiased.sh
