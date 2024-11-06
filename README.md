# Fake News Detection Model

This repository contains a fake news detection model developed using BERT and fine-tuned on a labeled dataset of real and fake news articles. 
The model classifies news articles as "Real" or "Fake."

​![pic](fake.jpg)

## Technical Overview

1. **Model Architecture**: BERT for sequence classification.
2. **Training Data**: A dataset of news articles labeled as "Fake" or "Real," preprocessed to extract text data.
3. **Fine-Tuning**: The BERT model was fine-tuned on an AWS SageMaker instance, with early stopping and regular checkpointing for resource efficiency.
4. **Evaluation**: The model's performance was assessed using precision, recall, and F1-score metrics.

## Repository Structure

- `train_script.py`: Script to train the model using SageMaker.
- `test_model.py`: Script to classify new text.
- `fake_news_model/`: Directory containing the saved model and tokenizer files.


## Dependencies
- Transformers (pip install transformers)
- PyTorch (pip install torch)

### Training
To train the model, run `train_script.py` in SageMaker with the specified parameters.

### Testing
To test new text:
```bash
python test_model.py
```
