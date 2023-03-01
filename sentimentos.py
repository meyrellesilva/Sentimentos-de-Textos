import requests

text = 'Este filme é muito bom'
response = requests.post('http://localhost:5000/', json={'text': text})
label = response.json()['label']
print(label)  # 'positive'



from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Inicialização do Flask
app = Flask(__name__)

# Carregamento do modelo e do vocabulário
model = pickle.load(open('model.pkl', 'rb'))
vocab = pickle.load(open('vocab.pkl', 'rb'))

# Pré-processamento de texto
def preprocess(text):
    stop_words = set(stopwords.words('portuguese'))
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    text = ' '.join(tokens)
    return text

# Rota principal
@app.route('/', methods=['POST'])
def classify():
    # Obtenção do texto a ser classificado
    text = request.json['text']

    # Pré-processamento do texto
    text = preprocess(text)

    # Criação do vetor de características
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform([text]).toarray()

    # Classificação do texto
    label = model.predict(X)[0]

    # Retorno da classificação
    return jsonify({'label': label})

# Execução do Flask
if __name__ == '__main__':
    app.run(debug=True)
