import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag_sents
import numpy as np
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('gutenberg')
import spacy
import os
import pandas as pd
os.system("python -m spacy download en_core_web_sm")
import en_core_web_sm
import collections
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def stats_pos(csv_file_path):
    data = pd.read_csv(csv_file_path)
    
    # Filter unique questions
    unique_q = data.drop_duplicates(subset=['qid'])['question']
    
    # Concat question
    concatenated_q = " ".join(unique_q)
    
    # Filter & concat answer sentences
    a_sentences = data[data['label'] == 1]['sentence text']
    concatenated_a = " ".join(a_sentences)
    
    # Tokenize & tag Qs
    q_sents = [word_tokenize(sent) for sent in sent_tokenize(concatenated_q)]
    q_tags = pos_tag_sents(q_sents, tagset='universal')
    
    # Tokenize & tag As
    a_sents = [word_tokenize(sent) for sent in sent_tokenize(concatenated_a)]
    a_tags = pos_tag_sents(a_sents, tagset='universal')
    
    # Flattening our tag list
    q_flat_tags = [tag for sent in q_tags for _, tag in sent]
    a_flat_tags = [tag for sent in a_tags for _, tag in sent]
    
    # Counting and normalizing frequencies
    q_tag_count = Counter(q_flat_tags)
    a_tag_count = Counter(a_flat_tags)
    
    total_question_tags = sum(q_tag_count.values())
    total_answer_tags = sum(a_tag_count.values())
    
    # Calculating normalized frequencies and rounding (4 Decimal places)
    q_freq = [(tag, round(count / total_question_tags, 4)) for tag, count in sorted(q_tag_count.items())]
    a_freq = [(tag, round(count / total_answer_tags, 4)) for tag, count in sorted(a_tag_count.items())]

    return q_freq, a_freq


def stats_top_stem_ngrams(csv_file_path, n, N):

    #Loading data
    df = pd.read_csv(csv_file_path)
    
    # Filter to unique questions and answers with label 1
    unique_q = df.drop_duplicates(subset='qid')
    answers = df[df['label'] == 1]

    # Concat texts
    all_q = ' '.join(unique_q['question'])
    all_a = ' '.join(answers['sentence text'])

    # function to calculate n-grams
    def get_ngrams(text, n):
        sentences = sent_tokenize(text)
        stemmer = PorterStemmer()
        ngrams = []
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            stems = [stemmer.stem(word.lower()) for word in words]
            if len(stems) >= n:
                ngrams.extend(list(nltk.ngrams(stems, n)))
                
        return ngrams
    
    # Get n-grams
    q_ngrams = get_ngrams(all_q, n)
    a_ngrams = get_ngrams(all_a, n)
    
    # Calculate and normalize frequencies
    q_freq = Counter(q_ngrams)
    a_freq = Counter(a_ngrams)
    total_q = sum(q_freq.values())
    total_a = sum(a_freq.values())

    # Calculate normalized frequency rounded to four decimals
    top_q_ngrams = [(ngr, round(freq / total_q, 4)) for ngr, freq in q_freq.most_common(N)]
    top_a_ngrams = [(ngr, round(freq / total_a, 4)) for ngr, freq in a_freq.most_common(N)]

    return top_q_ngrams, top_a_ngrams


def stats_ne(csv_file_path):
    
    # Load CSV
    data = pd.read_csv(csv_file_path)
    
    # Load spaCy's small English NER model
    nlp = spacy.load('en_core_web_sm')
    
    # Extract unique Q & A sentences based on label
    questions = data['question'].unique()  # Each question appears multiple times, process each unique one
    a_sentences = data[data['label'] == 1]['sentence text']  # Sentences that are part of the answer
    
    # Find the max length of questions to standardise the length of answer processing
    max_q_length = max(len(question) for question in questions)
    
    # # Count named entities in questions using spaCy
    q_entities = Counter()
    for question in questions:
        doc = nlp(question)
        q_entities.update([ent.label_ for ent in doc.ents])
    
    # Count named entities in answers, 
    a_entities = Counter()
    for sentence in a_sentences:
        truncated_sentence = sentence[:max_q_length] #limiing to the max question length
        doc = nlp(truncated_sentence)
        a_entities.update([ent.label_ for ent in doc.ents])
    
    # Calculate total entities for normalization
    total_q_entities = sum(q_entities.values())
    total_a_entities = sum(a_entities.values())
    
    # Normalize and sort the results and rounding
    normalized_q_entities = [(ent, round(freq / total_q_entities, 4)) for ent, freq in sorted(q_entities.items())]
    normalized_a_entities = [(ent, round(freq / total_a_entities, 4)) for ent, freq in sorted(a_entities.items())]
    
    return normalized_q_entities, normalized_a_entities

def stats_tfidf(csv_file_path):
    # Loading data
    data = pd.read_csv(csv_file_path)
    
    # Group by 'qid' so we can match each question with its answers
    grouped = data.groupby('qid')
    questions = []
    answers = []
    for name, group in grouped:
        questions.append(group['question'].iloc[0])  # Take the first occurrence of the question
        answers.append(group[group['label'] == 1]['sentence text'].tolist())  # All correct answers
    
    # combining all questions and sentences for TF-IDF analysis
    all_sentences = data['sentence text'].unique()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(questions + list(all_sentences))
    
    # Split TF-IDF vectors for questions and sentences
    questions_tfidf = tfidf_matrix[:len(questions)]
    sentences_tfidf = tfidf_matrix[len(questions):]
    
    # Cosine similarity between each question and all sentences
    cosine_similar = cosine_similarity(questions_tfidf, sentences_tfidf)
    
    # Most similar sentence for each question and if it's a correct answer
    count_correct = 0
    for idx, similarities in enumerate(cosine_similar):
        most_similar_idx = similarities.argmax()  # Index of the most similar sentence
        most_similar_sentence = all_sentences[most_similar_idx]
        # Check if this sentence is in the correct answers for the question
        if most_similar_sentence in answers[idx]:
            count_correct += 1
    
    # Calculate ratio and round to 4 decimals
    ratio = round(count_correct / len(questions), 4)
    
    return ratio


# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    # print("---------Task 1---------------")
    # print(stats_pos('data/dev_test.csv'))
  
    # print("---------Task 2---------------")
    # print(stats_top_stem_ngrams('data/dev_test.csv', 2, 5))

    # print("---------Task 3---------------")
    # print(stats_ne('data/dev_test.csv'))

    # print("---------Task 4---------------")
    # print(stats_tfidf('data/dev_test.csv'))
  
