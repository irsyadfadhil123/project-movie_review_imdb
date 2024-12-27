# 🌐 Sentiment Analysis of IMDb Comments

This repository contains a sentiment analysis project for IMDb comments. The project utilizes a machine learning model to classify comments as positive or negative, and includes training, deployment, and dataset details.

## 🔍 Overview

Analyzing user sentiments helps improve customer satisfaction and identify key issues. This project demonstrates how to preprocess data, train a sentiment analysis model, and deploy it using a Streamlit web interface.

## 📊 Dataset

The dataset used for training contains IMDb comments with labeled sentiments. Key features include:
- **Comment Text**: The text of the user comment.
- **Sentiment Label**: Binary sentiment labels (1 for positive, 0 for negative).

The dataset is designed to represent diverse sentiments for robust model training.

## 🛠️ Project Workflow

### 1. Training
- **Preprocessing**: Text cleaning, including emoticon removal, lowercase conversion, and punctuation stripping.
- **Tokenization**: Conversion of text into sequences using `Tokenizer`.
- **Padding**: Ensuring input sequences have uniform length.
- **Modeling**: Building a neural network model for classification.

### 2. Deployment
- **Streamlit Interface**: User-friendly interface for sentiment analysis.
- **Text Preprocessing**: On-the-fly cleaning and tokenization of user input.
- **Prediction**: Model inference using IBM Cloud API.

### 3. Model Evaluation
- Accuracy and other metrics to assess model performance during training.

## 🔄 How to Run

### Prerequisites
- Python 3.8+
- Required libraries (listed in `requirements.txt`):
  - numpy
  - pandas
  - tensorflow
  - streamlit
  - requests
  - pickle

### Training
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-imdb.git
   cd sentiment-analysis-imdb
   ```
2. Open the Jupyter Notebook for training:
   ```bash
   jupyter notebook training_GEA01I008.ipynb
   ```
3. Follow the steps in the notebook to train the model and save the tokenizer.

### Deployment
1. Install Streamlit and dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the deployment script:
   ```bash
   streamlit run deployment_GEA01I008.py
   ```
3. Open the Streamlit interface in your browser and input text for analysis.

## 🎯 Results
- **Predicted Sentiment**: Displays whether the input comment is positive or negative.
- **Probability Scores**: Detailed probabilities for each class.

## 🔬 Future Improvements
- Incorporate additional classes (neutral sentiment).
- Deploy using a cloud service for scalability.
- Enhance preprocessing steps for better model performance.

## 🙏 Contributing
Contributions are welcome! Please fork the repository and submit a pull request with enhancements or bug fixes.

## 📚 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
Enjoy exploring the project and feel free to share feedback! 😊

