from transformers import pipeline

try:
    sentiment_pipeline = pipeline(
       "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"Error loading NLP model: {e}")
    sentiment_pipeline = None


def analyze_feedback_text(text: str) -> dict:
    """
    Analyzes user feedback text to determine sentiment.

    Args:
        text (str): The feedback text submitted by the user.

    Returns:
        dict: A dictionary containing the sentiment label and score.
    """
    if not sentiment_pipeline:
        return {
            "sentiment": "unknown",
            "sentiment_score": 0.0,
            "error": "NLP model not available"
        }

    try:
        analysis_results = sentiment_pipeline(text)
        result = analysis_results[0]
        return {
            "sentiment": result.get('label'),
            "sentiment_score": result.get('score')
        }
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return {
            "sentiment": "error",
            "sentiment_score": 0.0
        }