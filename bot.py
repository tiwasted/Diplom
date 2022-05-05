import random
import json

import nltk
import sklearn.feature_extraction.text

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, CallbackContext, Filters, MessageHandler


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


# Запуск Бота


def bot(ml, question):
    if ml:
        intent = get_intent_by_ml(question)
    else:
        intent = get_intent(question)

    if intent != 'Не удалось определить интент':
        return (random.choice(BOT_CONFIG['intents'][intent]['responses']))
    else:
        return (random.choice(BOT_CONFIG['falture_phrases']))


# Подключение Telegram

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    ml = True
    update.message.reply_text(bot(ml, update.message.text))


def main() -> None:
    """Start the bot."""
    updater = Updater("")

    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
