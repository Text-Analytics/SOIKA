import pandas as pd
import nltk
import re

from dostoevsky.tokenization import RegexTokenizer 
from google.colab import drive
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()
pd.options.plotting.backend = "plotly"
nltk.download('stopwords')
stop_words = stopwords.words('russian')

def clean_text(text):
  # Удаляем все символы, кроме русских букв и пробелов
  text = re.sub('[^А-ЯЁа-яё ]', '', text)
  # Приводим текст к нижнему регистру и разбиваем на слова по пробелам
  words = text.lower().split()
  # Удаляем стоп-слова из списка слов
  words = [word for word in words if word not in stop_words]
  # Возвращаем очищенный текст в виде строки, соединяя слова пробелами
  return ' '.join(words)

def lemmatize_text(text):
  # Разбиваем текст на слова по пробелам
  words = text.split()
  # Для каждого слова получаем его лемму с помощью метода normal_form объекта анализатора 
  lemmas = [morph.parse(word)[0].normal_form for word in words]
  # Возвращаем лемматизированный текст в виде строки, соединяя леммы пробелами 
  return ' '.join(lemmas)

def sentiment_analysis(df_input):
    # Обработка текста
    df_input = df_input.iloc[1:]
    df = pd.DataFrame()
    df['text'] = df_input['Текст'].fillna('')
    df['clean_text'] = df['text'].apply(clean_text)
    df['lemmatized_text'] = df['clean_text'].apply(lemmatize_text)

    # Анализ тональности
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    results = model.predict(df.lemmatized_text, k=2)

    sentiment_list = []
    for sentiment in results:
        sentiment_list.append(sentiment)

    neutral_list = []
    negative_list = []
    positive_list = []
    speech_list = []
    skip_list = []
    for sentiment in sentiment_list:
        neutral = sentiment.get('neutral')
        negative = sentiment.get('negative')
        positive = sentiment.get('positive')
        if neutral is None:
            neutral_list.append(0)
        else:
            neutral_list.append(sentiment.get('neutral'))
        if negative is None:
            negative_list.append(0)
        else:
            negative_list.append(sentiment.get('negative'))
        if positive is None:
            positive_list.append(0)
        else:
            positive_list.append(sentiment.get('positive'))
    df['Обращение'] = neutral_list
    df['Жалоба'] = negative_list
    df['Благодарность'] = positive_list

    df_max = pd.DataFrame()
    df_max['Обращение'] = df['Обращение']
    df_max['Жалоба'] = df['Жалоба']
    df_max['Благодарность'] = df['Благодарность']

    df_max = pd.DataFrame()
    df_max['Обращение'] = df['Обращение']
    df_max['Жалоба'] = df['Жалоба']
    df_max['Благодарность'] = df['Благодарность']

    df_max["max_column"] = df_max.idxmax(axis=1)
    df['tonality'] = df_max["max_column"]
    df.drop(columns = ['clean_text', 'lemmatized_text'], axis = 1, inplace=True)

    return df
