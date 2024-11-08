from flask import Flask, request, render_template
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Data Cleaning
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess the text."""
    # Remove special characters and lowercase text
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # Tokenize text and remove stop words
    tokens = [word for word in text.split() if word not in stop_words]
    # Lemmatize tokens
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])
    return cleaned_text

# Load and preprocess the dataset
data = pd.read_csv(r"C:\Users\91983\Desktop\fakereview\dataset.csv")

 # Replace with your file path
print("Column names in the dataset:", data.columns)

# Apply the cleaning function to the 'text_' column
data['cleaned_review'] = data['text_'].apply(clean_text)

# Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = tfidf.fit_transform(data['cleaned_review']).toarray()
y = data['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_review(text):
    """Predict if the review is fake or real."""
    cleaned_text = clean_text(text)
    tfidf_features = tfidf.transform([cleaned_text]).toarray()
    prediction = model.predict(tfidf_features)
    return "Fake Review" if prediction == 1 else "Real Review"

# Flask Routes

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form.get('review')
    prediction = predict_review(review_text)
    return render_template('index.html', prediction_text=f'The review is: {prediction}')

    
    # Call the prediction function
    prediction = predict_review(review_text)
    
    # Return the result to the user
    return render_template('index.html', prediction_text=f'The review is: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
