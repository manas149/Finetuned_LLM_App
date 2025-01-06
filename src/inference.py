from transformers import pipeline

def test_inference():
    model_path = "./models/fine_tuned_model"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_path)

    # Test Examples
    test_texts = [
        "This movie was fantastic! The acting was great, and the plot was gripping.",
        "I didn't like the movie at all. It was too slow and boring."
    ]

    for text in test_texts:
        result = sentiment_pipeline(text)
        print(f"Review: {text}\nSentiment: {result}\n")

if __name__ == "__main__":
    test_inference()
