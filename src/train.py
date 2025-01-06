from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load Dataset
def load_imdb_data():
    dataset = load_dataset("imdb")
    dataset["train"].to_csv("data/train.csv")
    dataset["test"].to_csv("data/test.csv")
    return dataset["train"].shuffle(seed=42).select(range(16)), dataset["test"].shuffle(seed=42).select(range(16))

# Tokenize Data
def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

def main():
    # Model and Tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Load Data
    train_data, test_data = load_imdb_data()
    tokenized_train = tokenize_data(train_data, tokenizer)
    tokenized_test = tokenize_data(test_data, tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./models/fine_tuned_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=10,
        logging_dir="./logs"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
    )

    # Train the Model
    trainer.train()
    model.save_pretrained("./models/fine_tuned_model")
    tokenizer.save_pretrained("./models/fine_tuned_model")

if __name__ == "__main__":
    main()
