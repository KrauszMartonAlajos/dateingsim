import numpy as np
import pandas as pd
import random
import re
from sklearn.metrics.pairwise import cosine_similarity

#from gensim import corpora = CANCER = RÁK
from gensim.models import KeyedVectors

path = 'C:/Users/xboxh/Desktop/datesim/dateingsim/GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(path, binary=True)


listBOYS = []
listGIRLS = []

KEY_PHRASES = [
    "love", "hate", "happy", "sad", "family", "friends", "school", "stress", "homework", "relationship",
    "dreams", "goals", "future", "memories", "regret", "anger", "fear", "joy", "conflict", "anxiety",
    "peace", "grief", "boredom", "happiness", "excitement", "disappointment", "hope", "change", "self",
    "friendship", "loneliness", "inspiration"
]

def readFile(txt, marker, result_list):
    with open(txt, "r", encoding="utf-8") as f:
        content = f.read()  
        entries = content.split(marker)  
        for entry in entries:
            if entry.strip():  
                result_list.append(entry.strip())

def count_uppercase_words_not_first(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 1:  
            for word in words[1:]: 
                if word.isupper():  
                    count += 1
    
    return count

def sentence_structure_complexity(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|;)', text)
    
    if not sentences:
        return 0

    clause_counts = [
        len(re.split(r'(?<=\w),\s*(?=\w)|(?<=\w)\.\s*(?=\w)|(?<=\w)\s(and|or)\s(?=\w)', sentence))
        for sentence in sentences
    ]

    avg_clause_count = np.mean(clause_counts)
    
    return avg_clause_count

def min_max_normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) * 99 + 1

def calculate_key_phrase_density(text):
    words = text.lower().split()
    word_count = len(words)
    
    key_phrase_counts = {phrase: words.count(phrase) for phrase in KEY_PHRASES}
    
    total_key_phrases = sum(key_phrase_counts.values())
    
    if word_count == 0:
        return 0
    return total_key_phrases / word_count



def extract_features(text):
    avg_word_length = np.mean([len(word) for word in text.split()]) 

    sentence_count = text.count('.') + text.count('!') + text.count('?')   

    words = text.split()                                                                                            
    unique_words = set(words)                                                                                       
    vocabulary_richness = len(unique_words) / len(words)   

    # Example Word2Vec similarities (Placeholder, replace with actual calculation)
    word2vec_similarity_1 = findThemeinText("school", text, model, similarity_threshold=0.7)
    word2vec_similarity_2 = findThemeinText("work", text, model, similarity_threshold=0.7)                         

    emotional_intensity = sum(1 for word in words if word.lower() in ["love", "hate", "happy", "sad", "awfull", "terribble","amazing","wonderfull"]) / len(words)  

    key_phrase_density = calculate_key_phrase_density(text)

    sentence_structure_complexity_score = sentence_structure_complexity(text)

    separation_count = text.count(',') 

    name_count = count_uppercase_words_not_first(text)

    return [
        avg_word_length,
        sentence_count,
        vocabulary_richness,
        word2vec_similarity_1,
        word2vec_similarity_2,
        emotional_intensity,
        key_phrase_density,
        sentence_structure_complexity_score,
        separation_count,
        name_count
    ]     

def findThemeinText(theme, text, keyed_vectors, similarity_threshold=0.7):
    score = 0
    words = re.findall(r'\b\w+\b', text.lower())
    
    score += words.count(theme.lower())
    
    for word in words:
        if word != theme.lower() and word in keyed_vectors and theme in keyed_vectors:
            similarity = keyed_vectors.similarity(theme, word)
            if similarity >= similarity_threshold:
                score += 1

    return score

def avragecalc(df):
    listB = []
    listG = []
    for index, row in df.iterrows():
        avg_score = row.mean() #átlag
        if index.startswith("GIRL"): 
            listG.append(avg_score)
        elif index.startswith("BOY"):  
            listB.append(avg_score)
    return listB, listG

def find_closest_pairs(BOYlistAVG, GIRLlistAVG):
    pairs = []
    
    # összes kombó
    for girl_index, girl_score in enumerate(GIRLlistAVG):
        for boy_index, boy_score in enumerate(BOYlistAVG):
            girl_id = f"GIRL{girl_index + 1}"
            boy_id = f"BOY{boy_index + 1}"
            score_difference = abs(girl_score - boy_score)
            pairs.append((girl_id, boy_id, girl_score, boy_score, score_difference))
    
    
    pairs.sort(key=lambda x: x[4])
    
    # 20 legjobb nem dupe pár kiválasztása
    selected_pairs = []
    paired_girls = set()
    paired_boys = set()
    
    for girl, boy, girl_score, boy_score, diff in pairs:
        if girl not in paired_girls and boy not in paired_boys:
            selected_pairs.append((girl, boy, girl_score, boy_score, diff))
            paired_girls.add(girl)
            paired_boys.add(boy)
        if len(selected_pairs) == 20:
            break
    

    for index, (girl, boy, girl_score, boy_score, diff) in enumerate(selected_pairs):
        print(f"Pair {index + 1}: {girl} - {boy} {girl_score:.2f} - {boy_score:.2f} = {diff:.2f}")

def main():
    readFile("diariesBOY.txt", "-BOY-", listBOYS)
    readFile("diariesGIRL.txt", "-GIRL-", listGIRLS)
    print(len(listBOYS))
    print(len(listGIRLS))
    
    # Collecting all features
    all_features = []
    all_identifiers = []

    for i, boy_entry in enumerate(listBOYS):
        features = extract_features(boy_entry)
        all_features.append(features)
        all_identifiers.append(f"BOY{i+1}")

    for i, girl_entry in enumerate(listGIRLS):
        features = extract_features(girl_entry)
        all_features.append(features)
        all_identifiers.append(f"GIRL{i+1}")

    # Convert features into DataFrame for better representation
    columns = [
        "Avg Word Length",
        "Sentence Count",
        "Vocabulary Richness",
        "Word2Vec 1",
        "Word2Vec 2",
        "Emotional Intensity",
        "Key Phrase Density",
        "Sentence Complexity",
        "Separation Count",
        "Name Count"
    ]
    
    df = pd.DataFrame(all_features, columns=columns, index=all_identifiers)

    # Normalize all features to the range 1-100
    for column in columns:
        min_val = df[column].min()
        max_val = df[column].max()
        df[column] = df[column].apply(lambda x: min_max_normalize(x, min_val, max_val))

    BOYlistAVG, GIRLlistAVG = avragecalc(df)

    find_closest_pairs(BOYlistAVG, GIRLlistAVG)

main()
