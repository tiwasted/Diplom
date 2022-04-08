import random
import nltk
import re


def normalize(text):  # Создаем функцию, которая удалит знаки препинания и приведет текст к нижнему регистру
    text = text.lower()
    # Удалять из текста знаки препинания с помощью "Regular Expressions"
    punctuation = r"[^\w\s]"
    return re.sub(punctuation, "", text)


def ismatching(text1, text2):  # Создаем функцию, которая посчитает похожи ли два текста
    text1 = normalize(text1)
    text2 = normalize(text2)
    distance = nltk.edit_distance(text1, text2)  # Посчитаем расстояние между текстами (настолько они отличаются)
    avetage_length = (len(text1) + len(text2)) / 2  # Посчитаем среднюю длину текстов
    return distance / avetage_length < 0.4


def getintent(text):  # Понимать намерение по тексту
    all_intents = BOT_CONFIG["intents"]
    for name, data in all_intents.items():  # Пройти по всем намерениям и положить название в name, остальное в
        # переменную data
        for example in data["examples"]:  # Пройти по всем примерам этого intent и положить текст в переменную example
            if ismatching(text, example):  # Если текст совпадает с примером
                return name


question = input()  # Вопрос, который мы задаем боту

database = [
    {
        "question": "Помоги мне",
        "answer": ["Сделай глубокий вдох и выдох", "чем я могу тебе помочь"],
    },
    {
        "question": "У меня депрессия",
        "answer": "Какой-нибудь практический совет",
    },
    {
        "question": "Как справиться с тревогой",
        "answer": "Добавить упражнение",
    }
]

for pair in database:
    if ismatching(question, pair["question"]) < 0.4:
        answer = random.choice(pair["answer"])
        print(answer)

BOT_CONFIG = {
    "intents": {  # Намерения пользователя
        "hello": {  # Поздороваться
            "examples": ["привет", "Добрый день", "Здравствуйте"],
            "responses": ["Привет", "Вас приветствует Mental Health Бот", "Добро пожаловать"],
        },
        "depression": {
            "examples": ["Что делать когда у меня депрессия", "Депрессия", "У меня депрессия"],
            "responses": ["Совет", "Вдох-Выдох", "Практический совет"],
        },
        "failure_phrases": ["Извените, я Вас не понимаю", "К сожелению я всего лишь бот", "Очень жаль, но я здесь "
                                                                                          "бессилен"]
    }
}
