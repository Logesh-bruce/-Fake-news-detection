#!/usr/bin/env python3
# fake_news_detector.py

0-1import pandas as pd 
0-2from sklearn.model_selection import train_test_split 
0-3from sklearn.feature_extraction.text import TfidfVectorizer 
0-4from sklearn.linear_model import PassiveAggressiveClassifier 
0-5from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import joblib
import argparse

0-6def load_data(path): 
    0-7df = pd.read_csv(path) 
    0-8# Expect columns: 'text' and 'label', with label values 'FAKE'/'REAL' or 0/1 
    0-9if df['label'].dtype != 'int': 
        0-10df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0}) 
    0-11return df['text'], df['label'] 

0-12def train_and_evaluate(texts, labels, model_out, vect_out): 
    0-13x_train, x_test, y_train, y_test = train_test_split( 
        0-14texts, labels, test_size=0.2, random_state=42 
    )
    0-15vect = TfidfVectorizer(stop_words='english', max_df=0.7) 
    0-16tfidf_train = vect.fit_transform(x_train) 
    0-17tfidf_test = vect.transform(x_test) 

    clf = PassiveAggressiveClassifier(max_iter=50)
    clf.fit(tfidf_train, y_train)

    y_pred = clf.predict(tfidf_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['REAL','FAKE'])

    print(f'Accuracy: {acc*100:.2f}%')
    print('Confusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(cr)

    joblib.dump(clf, model_out)
    joblib.dump(vect, vect_out)
    print(f'Model saved to "{model_out}", vectorizer to "{vect_out}"')

def main():
    0-18parser = argparse.ArgumentParser(description='Fake News Detection Pipeline') 
    0-19parser.add_argument('--data', required=True, help='Path to CSV dataset') 
    0-20parser.add_argument('--model_out', default='model.pkl', help='Output model filename') 
    0-21parser.add_argument('--vect_out', default='vectorizer.pkl', help='Output vectorizer filename') 
    0-22args = parser.parse_args() 

    texts, labels = load_data(args.data)
    train_and_evaluate(texts, labels, args.model_out, args.vect_out)

0-23if _name_ == '_main_': 
    main()