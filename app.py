import gradio as gr
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and train model
print("Loading dataset and training model...")
fake = pd.read_csv("Fake.csv", on_bad_lines='skip', engine='python')
real = pd.read_csv("True.csv", on_bad_lines='skip', engine='python')

fake["label"] = 1
real["label"] = 0

df = pd.concat([fake, real], ignore_index=True)
df['content'] = df['title'] + " " + df['text']
df['content'] = df['content'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(df['content'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model trained successfully!")

# Prediction function
def detect_fake_news(title, article_text):
    full_content = str(title) + " " + str(article_text)
    cleaned = clean_text(full_content)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0]

    if prediction == 1:
        return f"🚨 FAKE NEWS\nConfidence: {confidence[1]*100:.2f}%"
    else:
        return f"✅ REAL NEWS\nConfidence: {confidence[0]*100:.2f}%"

# Gradio app
app = gr.Interface(
    fn=detect_fake_news,
    inputs=[
        gr.Textbox(label="📰 News Title", placeholder="Paste the news headline here..."),
        gr.Textbox(label="📄 News Article Text", placeholder="Paste the full article content here...", lines=8)
    ],
    outputs=gr.Textbox(label="🤖 AI Verdict"),
    title="🔍 Fake News Detector AI",
    description="Paste any news title and article text — our AI will tell you if it's REAL or FAKE! Built with Machine Learning by a 3rd year IT student 💪",
    examples=[
        ["Trump Middle East envoy meets Netanyahu in Jerusalem", "The White House said on Monday that Jason Greenblatt, the U.S. special envoy for international negotiations, met Israeli Prime Minister Benjamin Netanyahu in Jerusalem to discuss the peace process."],
        ["BREAKING: Aliens land in New York shake hands with president", "Multiple sources confirm that extraterrestrial beings have landed in Times Square and are currently negotiating with world leaders about taking over Earth resources."]
    ]
)

app.launch()