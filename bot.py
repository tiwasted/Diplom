import random
import nltk
import json

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# def normalize(text):  # Создаем функцию, которая удалит знаки препинания и приведет текст к нижнему регистру
#     text = text.lower()
#     # Удалять из текста знаки препинания с помощью "Regular Expressions"
#     punctuation = r"[^\w\s]"
#     return re.sub(punctuation, "", text)
#
#
# def ismatching(text1, text2):  # Создаем функцию, которая посчитает похожи ли два текста
#     text1 = normalize(text1)
#     text2 = normalize(text2)
#     distance = nltk.edit_distance(text1, text2)  # Посчитаем расстояние между текстами (настолько они отличаются)
#     avetage_length = (len(text1) + len(text2)) / 2  # Посчитаем среднюю длину текстов
#     return distance / avetage_length < 0.4
#
#
# def getintent(text):  # Понимать намерение по тексту
#     all_intents = BOT_CONFIG["intents"]
#     for name, data in all_intents.items():  # Пройти по всем намерениям и положить название в name, остальное в
#         # переменную data
#         for example in data["examples"]:  # Пройти по всем примерам этого intent и положить текст в переменную example
#             if ismatching(text, example):  # Если текст совпадает с примером
#                 return name
#
#
# def getanswer(intent):
#     responses = BOT_CONFIG["intents"][intent]["responses"]
#     return random.choice(responses)
#
#
# def bot(text):
#     intent = getintent(text)
#
#     if not intent:  # Если намерение не найдено
#         test = vectorizer.transform([text])
#         intent = model.predict(test)[0]
#
#     print("intent =", intent)
#
#     if intent:  # Если намерение найдено - выдать ответ
#         return
#
#     failure_phrases = BOT_CONFIG['failure_phrases']
#     return random.choice(failure_phrases)
# # question = input()  # Вопрос, который мы задаем боту
# #
# # database = [
# #     {
# #         "question": "Помоги мне",
# #         "answer": ["Сделай глубокий вдох и выдох", "чем я могу тебе помочь"],
# #     },
# #     {
# #         "question": "У меня депрессия",
# #         "answer": "Какой-нибудь практический совет",
# #     },
# #     {
# #         "question": "Как справиться с тревогой",
# #         "answer": "Добавить упражнение",
# #     }
# # ]
# #
# # for pair in database:
# #     if ismatching(question, pair["question"]) < 0.4:
# #         answer = random.choice(pair["answer"])
# #         print(answer)
#
#
# # BOT_CONFIG = {
# #     "intents": {  # Намерения пользователя
# #         "hello": {  # Поздороваться
# #             "examples": ["привет", "Добрый день", "Здравствуйте"],
# #             "responses": ["Привет", "Вас приветствует Mental Health Бот", "Добро пожаловать"],
# #         },
# #         "depression": {
# #             "examples": ["Что делать когда у меня депрессия", "Депрессия", "У меня депрессия"],
# #             "responses": ["Совет", "Вдох-Выдох", "Практический совет"],
# #         },
# #         "failure_phrases": ["Извените, я Вас не понимаю", "К сожелению я всего лишь бот", "Очень жаль, но я здесь "
# #                                                                                           "бессилен"]
# #     }
# # }
#
# config_file = open("big_bot_config.json", "r")
# BOT_CONFIG = json.load(config_file)
# X = []
# y = []
# for name, data in BOT_CONFIG["intents"].items():
#     for example in data['examples']:
#         X.append(example)
#         y.append(name)
#
#
# vectorizer = CountVectorizer()  # Можно указать настройки
# vectorizer.fit(X)  # Передаем набор текстов, чтобы векторайзер их проанализировал
#
# CountVectorizer()
#
# X_vectorized = vectorizer.transform(X)  # Трансформирует текст в вектор (наборы чисел)
#
# model = LogisticRegression()  # Настройки
# model.fit(X_vectorized, y)  # Модель научиться по X определять y
#
#
# BOT_KEY = '5042422719:AAGDb3wzvSdPHWk-bcW_tlnX704SMnmhb28'
#
#
# def hello(update: Update, context: CallbackContext) -> None:
#     update.message.reply_text(f'Hello {update.effective_user.first_name}')
#
#
# def botMessage(update: Update, context: CallbackContext):
#     text = update.message.text
#     reply = bot(text)
#     update.message.reply_text(reply)
#
#
# updater = Updater(BOT_KEY)
#
# updater.dispatcher.add_handler(CommandHandler('hello', hello))
# updater.dispatcher.add_handler(MessageHandler(Filters.text, botMessage))
#
# updater.start_polling()

with open('BOT_CONFIG.json', 'r') as f:
    BOT_CONFIG = json.load(f)


def clean(text):
    clean_text = ''
    for ch in text.lower():
        if ch in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
            clean_text = clean_text + ch
        return clean_text


# def get_intent(text):
#     for intent in BOT_CONFIG['intents'].keys():
#         for example in BOT_CONFIG['intents'][intent]['examples']:
#             cleaned_example = clean(example)
#             cleaned_text = clean(text)
#             if nltk.edit_distance(cleaned_example, cleaned_text) / max(len(cleaned_example), len(cleaned_text)) * 100 < 40:
#                 return intent
#     return 'unknown_intent'


# ОБУЧЕНИЕ МОДЕЛИ
X = []
y = []
for intent in BOT_CONFIG['intents'].keys():
    for example in BOT_CONFIG['intents'][intent]['examples']:
        X.append(example)
        y.append(intent)
len(X), len(y)


train_X, test_X, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 3))
X_train = vectorizer.fit_transform(train_X)
X_test = vectorizer.transform(test_X)
vocab = vectorizer.get_feature_names_out()
len(vocab)

clf = RandomForestClassifier(n_estimators=300).fit(X_train, X_test)
clf.score(X_train, y_train), clf.score(X_test, y_test)


def get_intent_by_model(text):
    vectorized_text = vectorizer.transform([text])
    return clf.predict(vectorized_text)[0]


def bot(text):
    intent = get_intent_by_model(text)

    # if intent == 'unknown_intent':
    #     return random.choice(BOT_CONFIG['default'])
    # else:
    return random.choice(BOT_CONFIG['intents'][intent]['responses'])


while True:
    input_text = input()
    response = bot(input_text)
    print(response)
