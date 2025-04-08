import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Page Configuration
st.set_page_config(page_title="AI Text Detector", page_icon=":guardsman:", layout="wide")

# Page Title
st.title("AI Text Detector")

# Sidebar Configuration
st.sidebar.title("Navigation")
st.sidebar.markdown("Use this sidebar to navigate the app")

# Sidebar Options
options = ["Home", "Train Model"]
selection = st.sidebar.selectbox("Go to", options)

if selection == "Home":
    st.header("Home")
    st.write("Welcome to the AI Text Detector!")

elif selection == "Train Model":
    st.header("Train Model")
    st.write("Train a model to detect AI-generated text.")

    # Sample data (replace this with actual dataset loading)
    data = {
        'text': [
            "This is a sample human-written text.",
            "GPT-3 generated text goes here.",
            "Another example of human-written content.",
            "More AI-generated content follows."
        ],
        'label': [0, 1, 0, 1]  # 0 for human-written, 1 for AI-generated
    }

    df = pd.DataFrame(data)

    # Ensure the labels are correctly indexed
    df['label'] = df['label'].astype(int)

    # Train-Test Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    # Custom Dataset
    class CustomDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    # Load Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
    )

    # Trainer
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Training
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()
    st.write(f"Evaluation Result: {eval_result}")

    # Save the model and tokenizer
    model_save_path = './model'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    st.write(f"Model and tokenizer saved in {model_save_path}")
