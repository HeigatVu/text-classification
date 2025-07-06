# Spam/Ham SMS Classification using Naïve Bayes

This project is a machine learning application that classifies SMS messages as either "spam" (unwanted promotional or malicious messages) or "ham" (normal, desired messages). The classification is performed using a Naïve Bayes classifier, a popular and effective algorithm for text classification tasks.

## 📝 Project Overview

The goal of this project is to build and train a model that can accurately distinguish between spam and ham messages. The process involves several key stages of a standard machine learning pipeline: data preprocessing, feature engineering, model training, and evaluation.

The final model takes a text message as input and outputs a prediction of whether it is spam or ham.

### Pipeline
The project follows this pipeline:
1.  **Load Data**: Import the dataset containing messages and their corresponding labels.
2.  **Preprocess Text**: Clean and normalize the text data to prepare it for feature extraction.
3.  **Build Vocabulary**: Create a dictionary of all unique words from the preprocessed messages.
4.  **Create Features**: Convert the text messages into numerical vectors (Bag-of-Words).
5.  **Train Model**: Train a Gaussian Naïve Bayes classifier on the feature vectors.
6.  **Evaluate & Predict**: Assess the model's performance and use it to classify new, unseen messages.

![Pipeline Diagram](https://i.imgur.com/8aZ2a2o.png)

## 💾 Dataset

The model is trained on a publicly available dataset of SMS messages. The dataset consists of two columns:
* `Category`: The label for the message, which is either `ham` or `spam`.
* `Message`: The raw text content of the SMS message.

The dataset can be downloaded from [this Kaggle page](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

## ⚙️ Methodology

### 1. Data Preprocessing
To prepare the text for the model, each message undergoes the following preprocessing steps:
* **Lowercasing**: All text is converted to lowercase.
* **Punctuation Removal**: All punctuation marks are removed.
* **Tokenization**: The text is split into a list of individual words (tokens).
* **Stopword Removal**: Common words that carry little semantic meaning (e.g., "the", "a", "is") are filtered out.
* **Stemming**: Words are reduced to their root form (e.g., "studying" becomes "studi").

### 2. Feature Engineering
After preprocessing, each message is converted into a numerical vector using a **Bag-of-Words** approach. A vocabulary of all unique words across the entire dataset is created. Then, for each message, a feature vector is generated where each element represents the frequency count of a word from the vocabulary in that message.

### 3. Model Training
The dataset is split into three sets:
* **Training Set (70%)**: Used to train the model.
* **Validation Set (20%)**: Used to tune hyperparameters and evaluate the model during training.
* **Test Set (10%)**: Used for the final, unbiased evaluation of the model's performance.

A **Gaussian Naïve Bayes (GaussianNB)** classifier from the `scikit-learn` library is trained on the training set.

## 📈 Results

The trained model was evaluated on the validation and test sets, achieving the following accuracy scores:

| Data Set   | Accuracy |
| :--------- | :------- |
| Validation | 88.16%   |
| Test       | 86.02%   |

## 🛠️ Technologies Used

* **Python**
* **Google Colab**
* **Pandas**: For data manipulation and reading CSV files.
* **NumPy**: For numerical operations and creating feature vectors.
* **NLTK (Natural Language Toolkit)**: For text preprocessing tasks like tokenization, stopword removal, and stemming.
* **Scikit-learn**: For splitting the data, building the Naïve Bayes model, and evaluating its performance.

## 🚀 How to Run

The project is implemented in a Google Colab notebook. To run it, follow these steps:
1.  Open the `.ipynb` file in Google Colab.
2.  The notebook will automatically download the required dataset using the `gdown` command.
3.  Run the cells sequentially from top to bottom to execute the entire pipeline, from data loading to prediction.

### Example Prediction
To predict a new message, you can use the `predict` function defined in the notebook:

```python
# Example of classifying a new message
test_input = 'Free entry in 2 a wkly comp to win FA Cup final tkts'
prediction_cls = predict(test_input, model, dictionary)
print(f'Prediction: {prediction_cls}')
```
