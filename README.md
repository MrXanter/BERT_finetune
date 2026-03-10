# BERT Sentiment Analysis Fine-tuning

This project demonstrates how to fine-tune a BERT model for sentiment analysis using the Hugging Face Transformers library. The model is trained to classify text into three sentiment categories: Negative, Neutral, and Positive.

## Features

- Fine-tuning `bert-base-uncased` for 3-class sentiment classification
- Comprehensive training pipeline with evaluation metrics
- Model checkpointing and best model selection
- Visualization of training progress and model performance
- Easy inference for predicting sentiment of new text

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── finetune.ipynb      # Main training notebook
├── sentiment_dataset.csv  # Training data
├── train_dataset.csv   # Training split
├── eval_dataset.csv    # Evaluation split
├── results/           # Model checkpoints and outputs
├── logs/              # Training logs
├── requirements.txt   # Project dependencies
└── README.md          # This file
```

## Model Performance

The model achieves approximately 76% accuracy on 3-class sentiment classification. Performance metrics include:

- Accuracy
- F1-Score
- Precision
- Recall

## Inference Example

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-2500")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Predict sentiment
text = "I love this product!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1)
```

## Visualization

The project includes comprehensive visualization of:

- Training and evaluation metrics over time
- Model performance comparison across checkpoints
- Detailed performance breakdown by metric

## License

This project is open source and available under the MIT License.
