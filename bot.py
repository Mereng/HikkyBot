from NN import NN
import telebot
import config
import re

regex = re.compile(r'[^A-z\s]');
regex_command = re.compile(r'^\/[\w@]+\s')

nn = NN()
bot = telebot.TeleBot(config.token)


@bot.message_handler(content_types=['text'])
def answer(message):
    answer_nn = take_answer(message.text)
    bot.send_message(message.chat.id, answer_nn)


@bot.message_handler(commands=['say'])
def answer_to_group(message):
    answer_nn = take_answer(message.text)
    bot.send_message(message.chat.id, answer_nn)


def take_answer(message):
    message = regex_command.sub('', message)
    message = regex.sub('', message)
    message = message.lower()
    if len(message) != 0:
        return nn.take_answer(message)
    else:
        return 'I am not understand you'

if __name__ == '__main__':
    while True:
        try:
            bot.polling(none_stop=True)
        except:
            pass

