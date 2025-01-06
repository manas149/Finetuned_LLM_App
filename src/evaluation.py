import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# Load the fine-tuned model and tokenizer
model_path = "./models/fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load the test dataset
def load_test_data():
    dataset = load_dataset("imdb")
    return dataset["test"].shuffle(seed=42).select(range(100))

# Tokenize the dataset
def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

# Compute metrics
def compute_metrics(pred):
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    # Extract predictions and labels
    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids

    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")  # Returns a float
    precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")  # Returns a float
    recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")  # Returns a float

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# Evaluate the model
def evaluate_model():
    # Load and tokenize test data
    test_data = load_test_data()
    tokenized_test = tokenize_data(test_data, tokenizer)

    # Define TrainingArguments for evaluation
    training_args = TrainingArguments(
        output_dir="./models/fine_tuned_model",
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        do_train=False,
        do_eval=True
    )

    # Define Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Evaluate the model
    eval_result = trainer.evaluate()
    print(f"Evaluation Results: {eval_result}")

if __name__ == "__main__":
    evaluate_model()
