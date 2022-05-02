import random
import json

import nltk
import sklearn.feature_extraction.text

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


with open("BOT_CONFIG.json", "r", encoding="utf8") as file:
    BOT_CONFIG = json.load(file)


def clean(text):
    clean_text = ''
    for char in text.lower():
        if char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя":
            clean_text = clean_text + char
    return clean_text


def compare(s1, s2):
    return nltk.edit_distance(s1, s2) / ((len(s1) + len(s2)) / 2) < 0.4


def get_intent(question):
    for intent in BOT_CONFIG['intents']:
        for example in BOT_CONFIG['intents'][intent]['examples']:
            if compare(clean(example), clean(question)):
                return intent
    return 'Не удалось определить интент'

# Обучение модели


X = []
y = []
for intent in BOT_CONFIG['intents']:
    X += BOT_CONFIG['intents'][intent]['examples']
    y += [intent] * len(BOT_CONFIG['intents'][intent]['examples'])


len(X), len(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
len(X_train), len(X_test)


vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(1, 2), min_df=1)
vectorizer.fit(X_train)


X_train_vectorized = vectorizer.transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


classifier = RandomForestClassifier()
classifier.fit(X_train_vectorized, y_train)


def get_intent_by_ml(text):
    return classifier.predict(vectorizer.transform([text]))[0]


# print(model.predict(vectorizer.transform(["Депрессия"])))


# def get_intent(input_text):
#     for intent in BOT_CONFIG["intents"].keys():
#         for example in BOT_CONFIG["intents"][intent]["examples"]:
#             text1 = input_text.lower()
#             text2 = example.lower()
#             if edit_distance(text1, text2) / max(len(text1), len(text2)) < 0.34:
#                 return intent
#     return "Not Found"


# def bot(input_text):
#     intent = get_intent(input_text)
#     if intent == "Not found":
#         return "Мы вас не поняли"
#     else:
#         return random.choice(BOT_CONFIG["intents"][intent]["responses"])


def get_intent_by_ml(text):
    return model.predict(vectorizer.transform([text]))[0]

# Запуск Бота


def bot(ml):
    question = input()

    if ml:
        intent = get_intent_by_ml(question)
    else:
        intent = get_intent(question)

    if intent != 'Не удалось определить интент':
        print(random.choice(BOT_CONFIG['intents'][intent]['responses']))
    else:
        print(random.choice(BOT_CONFIG['falture_phrases']))


while True:
    bot(True)


# Подключение Telegram


def bot(ml, question):
    if ml:
        intent = get_intent_by_ml(question)
    else:
        intent = get_intent(question)

    if intent != 'Не удалось определить интент':
        print(random.choice(BOT_CONFIG['intents'][intent]['responses']))
    else:
        print(random.choice(BOT_CONFIG['falture_phrases']))
