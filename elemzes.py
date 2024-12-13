import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors

# GLOBAL VARIABLES
SEED = 42
np.random.seed(SEED)

# Word2Vec model betöltése
# (Cseréld ki egy helyi vagy távoli modellel, pl. Google News Word2Vec.)
# w2v_model = KeyedVectors.load_word2vec_format('path/to/word2vec.bin', binary=True)
w2v_model = None  # Példa helykitöltő

# 1. DATA STRUCTURE FOR FEATURES
columns = [
    "avg_word_length",  # Átlagos szóhossz
    "sentence_count",   # Mondatok száma
    "vocabulary_richness",  # Egyedi szavak aránya
    "sentiment_score",  # Szöveg szentiment elemzése
    "frequent_topics",  # Gyakori témák aránya
    "word2vec_similarity_1",  # Word2Vec alapján valamilyen hasonlóság
    "word2vec_similarity_2",  # Word2Vec egy másik aspektusa
    "emotional_intensity",  # Érzelmi szavak aránya
    "key_phrase_density",  # Kulcskifejezések aránya
    "dialogue_presence"   # Dialógus jelenléte a szövegben
]

def generate_empty_features(n):
    """Létrehoz egy üres DataFrame-et a feature-ek tárolására."""
    return pd.DataFrame(np.zeros((n, len(columns))), columns=columns)

# 2. TEXT PROCESSING PIPELINE
def extract_features(text, model=None):
    """Kivonja a szöveges feature-eket egy naplórészletből.
    Arguments:
        text (str): A naplórészlet szövege.
        model (KeyedVectors): Word2Vec modell a hasonlósági feature-ekhez.
    Returns:
        list: Kiszámított feature-értékek.
    """
    # Példa feature-ek kiszámítása
    avg_word_length = np.mean([len(word) for word in text.split()])
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    words = text.split()
    unique_words = set(words)
    vocabulary_richness = len(unique_words) / len(words)
    sentiment_score = 0  # Ideális esetben egy NLP sentiment modell használata

    # Word2Vec hasonlóságok (helykitöltők, valódi modellel cseréld ki)
    word2vec_similarity_1 = 0
    word2vec_similarity_2 = 0

    emotional_intensity = sum(1 for word in words if word.lower() in ["love", "hate", "happy", "sad"]) / len(words)
    key_phrase_density = 0.1  # Példaérték, számítsd ki valós kulcskifejezésekkel
    dialogue_presence = text.count('"') / max(sentence_count, 1)

    return [
        avg_word_length, sentence_count, vocabulary_richness, sentiment_score,
        0, word2vec_similarity_1, word2vec_similarity_2,
        emotional_intensity, key_phrase_density, dialogue_presence
    ]

# 3. SIMULATION OF FEATURE EXTRACTION
n_boys, n_girls = 20, 20
boy_texts = [f"Boy {i} diary text." for i in range(n_boys)]  # Helykitöltő szövegek
girl_texts = [f"Girl {i} diary text." for i in range(n_girls)]

# Feature-ek kiszámítása
boy_features = [extract_features(text, w2v_model) for text in boy_texts]
girl_features = [extract_features(text, w2v_model) for text in girl_texts]

boy_df = pd.DataFrame(boy_features, columns=columns)
girl_df = pd.DataFrame(girl_features, columns=columns)

# 4. MATCHING PAIRS USING NAIVE BAYES
# Összefésülés egyetlen DataFrame-be
boy_df["label"] = "boy"
girl_df["label"] = "girl"
all_data = pd.concat([boy_df, girl_df], ignore_index=True)

# Adatok szétválasztása tanítási és tesztelési szettekre
X = all_data[columns]
y = all_data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

# Naiv Bayes osztályozó betanítása
model = MultinomialNB()
model.fit(X_train, y_train)

# Predikció
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2f}")

# 5. PAIR OPTIMIZATION
def optimize_pairs(boy_features, girl_features):
    """Optimalizálja a fiú-lány párokat a feature-ek alapján."""
    boy_indices = list(range(len(boy_features)))
    girl_indices = list(range(len(girl_features)))
    
    pairs = []
    while boy_indices and girl_indices:
        b_idx = boy_indices.pop(0)
        g_idx = girl_indices.pop(0)
        pairs.append((b_idx, g_idx))
    
    return pairs

pairs = optimize_pairs(boy_features, girl_features)
print("Optimized pairs:", pairs)
