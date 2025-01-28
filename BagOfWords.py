import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#Load spaCy model
nlp = spacy.load("en_core_web_sm")

#Amazon reviews dataset
amazon_reviews = [
    "I love this product! It's amazing and works perfectly. Highly recommend it to everyone.",
    "The quality is good, but the delivery was late. Still, I'm happy with my purchase.",
    "Terrible product. It broke within a week. Never buying from this brand again.",
    "The customer service is outstanding. They helped resolve my issue very quickly.",
    "This product is just okay. Nothing special, but not bad either. Fair for the price."
]

#Sentence Tokenization
print("\nStep 4: Sentence Tokenization\n")
all_sentences = []
for review in amazon_reviews:
    doc = nlp(review)
    sentences = [sent.text for sent in doc.sents]
    all_sentences.extend(sentences)
    print(f"Review: {review}")
    print(f"Sentences: {sentences}\n")

#Feature Extraction - Bag of Words (BoW)
print("\nStep 5: Feature Extraction - Bag of Words (BoW)\n")
vectorizer_bow = CountVectorizer()
bow_matrix = vectorizer_bow.fit_transform(amazon_reviews)
print("Vocabulary (BoW):")
print(vectorizer_bow.get_feature_names_out())
print("\nBag of Words Matrix:")
print(pd.DataFrame(bow_matrix.toarray(), columns=vectorizer_bow.get_feature_names_out()))
