import random
import json

from nltk import edit_distance

from sklearn.feature_extraction.text import CountVectorizer


with open("BOT_CONFIG.json", "r") as file:
    BOT_CONFIG = json.load(file)


X = []
y = []

count = 0

for intent in BOT_CONFIG["intents"].keys():
    try:
        for example in BOT_CONFIG["intents"][intent]["examples"]:
            X.append(example)
            y.append(intent)
    except KeyError:
        print(BOT_CONFIG["intents"][intent])


def clean_text(text):
    output_text = ""
    for char in text:
        if char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя":
            output_text = output_text + char
    return output_text


def get_intent(input_text):
    for intent in BOT_CONFIG["intents"].keys():
        for example in BOT_CONFIG["intents"][intent]["examples"]:
            text1 = input_text.lower()
            text2 = example.lower()
            if edit_distance(text1, text2) / max(len(text1), len(text2)) < 0.34:
                return intent
    return "Not Found"


def bot(input_text):
    intent = get_intent(input_text)
    if intent == "Not found":
        return "Мы вас не поняли"
    else:
        return random.choice(BOT_CONFIG["intents"][intent]["responses"])


input_text = ""
print("Для выхода из диалога напишите Пока")

while True:
    input_text = input()
    if input_text == "Пока":
        break
    print(bot(input_text))
