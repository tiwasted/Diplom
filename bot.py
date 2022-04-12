import random
import json

import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


with open("BOT_CONFIG.json", "r", encoding="utf8") as file:
    BOT_CONFIG = json.load(file)


# X = []
# y = []
#
# count = 0
#
# for intent in BOT_CONFIG["intents"].keys():
#     try:
#         for example in BOT_CONFIG["intents"][intent]["examples"]:
#             X.append(example)
#             y.append(intent)
#     except KeyError:
#         print(BOT_CONFIG["intents"][intent])


X = []
y = []
for intent in BOT_CONFIG['intents']:
    X += BOT_CONFIG['intents'][intent]['examples']
    y += [intent] * len(BOT_CONFIG['intents'][intent]['examples'])


vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)


model = LogisticRegression(random_state=42)
model.fit(X_vectorized, y)


# print(model.predict(vectorizer.transform(["Депрессия"])))


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


# input_text = ""
# print("Для выхода из диалога напишите Stop")
#
# while True:
#     input_text = input()
#     if input_text == "Stop":
#         break
#     print(bot(input_text))
