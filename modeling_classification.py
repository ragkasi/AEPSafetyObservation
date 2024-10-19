from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from torch.utils.data import Dataset

# Custom Dataset class to handle inputs and labels
class SafetyObservationDataset(Dataset):
    def __init__(self,encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self,idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Load the pre-trained BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input text
def tokenize_data(data):
    print(data)
    return tokenizer(data.tolist(), padding=True, truncation=True, return_tensors="pt")

# Fine-tune BERT model
def fine_tune_model(X_train, y_train):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    # Prepare training data
    # inputs = tokenize_data(X_train)
    # labels = torch.tensor(y_train.values)
    encodings = tokenize_data(X_train)
    dataset = SafetyObservationDataset(encodings, y_train.values)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir = './results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy = "epoch",
        save_total_limit=2, # Save the last two models
        fp16 = True
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=inputs
    )

    # Train the model
    trainer.train()

    return model

# Predict risk level for test data
def classify_comments(model,X_test):
    inputs = tokenize_data(X_test)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions

if __name__ == "__main__":
    # Load the preprocessed train and test data
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')

    # Assuming the 'risk_level' column exist and labels data as 'low', 'medium', or 'high'
    X_train,y_train = train_data['clean_comments'], train_data['risk_level']
    X_test = test_data['clean_comments']

    # Fine-tune the model on the training data
    model = fine_tune_model(X_train, y_train)


    # Classify test data
    predictions = classify_comments(model, X_test)
    test_data['predicted_risk_level'] = predictions
    test_data.to_csv('classified_test_data.csv', index=False)
