import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src import preprocessing


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
MODEL_PATH = MODEL_DIR / 'best_optimized_bilstm.h5'
TOKENIZER_PATH = MODEL_DIR / 'tokenizer.json'




@st.cache_resource
def load_tokenizer(tokenizer_path: str = str(TOKENIZER_PATH)):
if not Path(tokenizer_path).exists():
st.error('Tokenizer not found. Please train the model first (run src/train.py).')
return None
with open(tokenizer_path, 'r', encoding='utf-8') as f:
data = f.read()
tokenizer = tokenizer_from_json(data)
return tokenizer




@st.cache_resource
def load_sentiment_model(model_path: str = str(MODEL_PATH)):
if not Path(model_path).exists():
st.error('Model file not found. Please train the model first (run src/train.py).')
return None
model = load_model(model_path, compile=False)
# compile with same settings as training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
return model




def predict_sentiment(text: str, model, tokenizer, max_len: int = 100):
seq = tokenizer.texts_to_sequences([text])
seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
preds = model.predict(seq)
label = int(np.argmax(preds, axis=-1)[0])
return label, preds




# Minimal Streamlit UI
st.title('Sentiment — Airline')


tokenizer = load_tokenizer()
model = load_sentiment_model()


tweet = st.text_area('Enter text to analyze')
if st.button('Analyze Sentiment'):
if not tweet.strip():
st.warning('Please enter text to analyze')
elif model is None or tokenizer is None:
st.error('Model or tokenizer missing — train the model first')
else:
label, probs = predict_sentiment(tweet, model, tokenizer, max_len=100)
st.success(f'Predicted label: {label} — probs: {probs[0].tolist()}')