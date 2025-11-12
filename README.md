# ✈️ Sentiment Analysis on Airline Tweets

This project classifies airline-related tweets into **Positive**, **Neutral**, or **Negative** sentiments using a **Deep BiLSTM + Attention** model with **GloVe embeddings**.

## 🧠 Key Features
- Deep BiLSTM + Attention Architecture
- Pre-trained GloVe 200d Word Embeddings
- Modular Pipeline: Preprocessing → Training → Evaluation → Deployment
- Streamlit UI for Real-Time Tweet Classification

## ⚙️ Setup
```bash
git clone https://github.com/your-username/Sentiment-Analysis-Airline-Tweets.git
cd Sentiment-Analysis-Airline-Tweets
pip install -r requirements.txt
streamlit run src/app.py
