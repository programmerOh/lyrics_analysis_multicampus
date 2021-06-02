from os import PRIO_USER
import re
import csv
from konlpy.tag import Mecab
from numpy.core.records import array
from emotion_dictionary_kr import load_dictionary_on_emotion
import pandas as pd
import numpy as np

# df_emo = load_dictionary_on_emotion()

mecab = Mecab()
# mecab 사용자 사전
# komoran 사용자 사전 (기분석 사전)


def load_dictionary_on_song():
    big_list = []# 파일로 출력하기 위해
    with open('./first_project/Crawled_Updated.csv') as books_csv:
        books_reader = csv.DictReader(books_csv)
        for row in books_reader:
            test = {}
            big_list.append(test)
            for k,v in row.items():
                if k=='songid' or k=='artist':
                    test.update({k:v})
                if k=='lyrics':
                    test.update({'morphs':mecab.morphs(v)})
                    test.update({'pos':mecab.pos(v)})
                    test.update({'nouns':mecab.nouns(v)})
    return big_list
big_list = load_dictionary_on_song()
# print(df_lyrics)
def save_and_load():
    with open('output.csv', 'w') as output_csv:
        fields = ['songid', 'artist', 'pos', 'morphs', 'nouns']
        output_writer = csv.DictWriter(output_csv, fieldnames=fields)
        output_writer.writeheader()
        for row in big_list:
            output_writer.writerow(row)


    with open('output.csv') as input_csv:
        reader = csv.DictReader(input_csv, delimiter=',')
        result_list = []
        for row in reader:
            result_list.append(row)
        df_lyrics = pd.DataFrame(result_list)

    return df_lyrics
# 불용어(Stopword)
# 토큰화 후에 조사, 접속사 등을 제거하는 방법이 있습니다. 
# 하지만 불용어를 제거하려고 하다보면 조사나 접속사와 같은 단어들뿐만 아니라 
# 명사, 형용사와 같은 단어들 중에서 불용어로서 제거하고 싶은 단어들이 생기기도 합니다. 
# 결국에는 사용자가 직접 불용어 사전을 만들게 되는 경우가 많습니다
# 코드 작성필요 << https://www.ranks.nl/stopwords/korean << mecab을 통해 POS 정보 부여가능?

def get_stopwords():
    stop_words = []
    return stop_words
stop_words = get_stopwords()

def indexing(raw_data, feature_num, stop_words_param=[]):
    # 전체 Data에 대한 분석 제공
    vocab={}# 정수 인코딩(Integer Encoding)에 사용할 예정
    # 예: vocab에는 중복을 제거한 단어와 각 단어에 대한 빈도수가 기록

    # bag-of-words dictionary: this dictionary is comprised of words with their corresponding counts
    for tokenized in raw_data:
        for word in tokenized:
            # 불용어(Stopword)
            # 토큰화 후에 조사, 접속사 등을 제거하는 방법이 있습니다. 
            #  +
            # 정수 인코딩(Integer Encoding) >> 워드 임베딩 챕터(Word Embedding) 등에서 사용
            # 인덱스(고유한 정수)를 부여하는 방법은 여러 가지가 있을 수 있는데 랜덤으로 부여하기도 하지만, 
            # 보통은 전처리 또는 빈도수가 높은 단어들만 사용하기 위해서 단어에 대한 빈도수를 기준으로 정렬한 뒤에 부여합니다.
            if word not in stop_words_param:
                if word not in vocab:
                    # python의 Counter 함수, NLTK의 FreqDist 함수를 사용할경우 각 단어별 빈도수와 최소빈도를 지정할수 있음
                    # 하지만 형태소의 분류에 따른 구분은 안됨(단어별)
                    vocab[word] = 0
                vocab[word] += 1
    # 현재 vocab에는 중복을 제거한 단어와 각 단어에 대한 빈도수가 기록
    # 빈도수가 높은 순서대로 정렬
 
    vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
    # 이제 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여
    word_to_index = {}#feachires dictionary : each word is mapper to a vector index
    i=0
    for (word, frequency) in vocab_sorted :
        if frequency > 10 : # 정제(Cleaning) 챕터에서 언급했듯이 빈도수가 적은 단어는 제외, 등장 빈도가 낮은 단어는 자연어 처리에서 의미를 가지지 않을 가능성이 높기 때문
            i=i+1# 첫 index가 1부터 시작, 0은 나중에 matrix 형성시 사용할 값
            #{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7}
            # 이때 word가 tuple이 사용될수 있는지 확인필요 
            word_to_index[word] = [i, frequency] 
    # 빈도수 기준으로 정렬후 상위 n개의 단어만 사용하는 경우
    vocab_size = feature_num # 고려할 변수의 수
    words_frequency = [w for w,c in word_to_index.items() if c[0] >= vocab_size + 1] # 인덱스가 vocab_size 초과인 단어 제거
    for w in words_frequency:
        del word_to_index[w] # 해당 단어에 대한 인덱스 정보를 삭제
    # 단어 집합에 존재하지 않는 단어들을 Out-Of-Vocabulary(단어 집합에 없는 단어)의 약자로 'OOV'라고 합니다. 
    # word_to_index에 'OOV'란 단어를 새롭게 추가하고, 단어 집합에 없는 단어들은 'OOV'의 인덱스로 인코딩하겠습니다.
    word_to_index[('OOV','-')] = [len(word_to_index) + 1, np.sum([frequency for (word, frequency) in vocab_sorted if word not in word_to_index.keys()])]

    # return word_to_index, vocab_sorted, raw_data
    df = pd.DataFrame([[k[0],k[1],v[0],v[1],k] for k, v in word_to_index.items()])
    return df, raw_data
# 

# df_lyrics = save_and_load()
# grouped = df_lyrics.groupby(['artist']).count().sort_values(by=['songid'], ascending=False)
# word_sort_by_artist = {}
# for art in grouped.index:
#     word_to_index, vocab, raw_data = indexing(df_lyrics[df_lyrics['artist'] == art]['pos'], stop_words_param=stop_words)
#     df1 = pd.DataFrame([(key, value) for key, value in word_to_index.items()])
#     df2 = pd.DataFrame(vocab)
#     merged = df1.merge(df2, how='left',left_on=0, right_on=0).rename(columns={0:'word', '1_x':'rank', '1_y':'count'})
#     merged['pos'] = merged['word'].apply(lambda x: list(x)[1])
#     merged['word'] = merged['word'].apply(lambda x: list(x)[0])
#     word_sort_by_artist.update({art:merged})
# print(word_sort_by_artist)


def integer_encoding(word_to_index, raw_data):
    # 모든 단어들을 맵핑되는 정수로 인코딩
    encoded_lyrics = []
    for tokenized in raw_data:
        indexed_lyrics = []#한곡
        encoded_lyrics.append(indexed_lyrics)#여러곡
        for word in tokenized:
            if word_to_index.get(word) is not None:
                indexed_lyrics.append(word_to_index.get(word))
            else:
                indexed_lyrics.append(word_to_index.get('OOV'))
    #자연어 처리를 하다보면 각 문장(또는 문서)은 서로 길이가 다를 수 있습니다. 
    #그런데 기계는 길이가 전부 동일한 문서들에 대해서는 하나의 행렬로 보고, 한꺼번에 묶어서 처리할 수 있습니다. \\
    #다시 말해 병렬 연산을 위해서 여러 문장의 길이를 임의로 동일하게 맞춰주는 작업이 필요할 때가 있습니다.
    # padding 가장 긴것을 기준으로 할필요는 없음, 속성 길이
    num_on_features = max(len(lyrics) for lyrics in encoded_lyrics)
    # 가장 긴 길이를 사용하지 않을경우 길이가 초과되는 row의 값들을 잘라내는 기준이 필요
    # num_on_features = np.percentile(len(lyrics) for lyrics in encoded_lyrics, 75)
    for lyrics in encoded_lyrics:
        while len(lyrics) < num_on_features:
            lyrics.append(0)
    padded_np = np.array(encoded_lyrics)    
    # 길이가 짧은 문장에는 전부 숫자 0이 뒤로 붙어서 모든 문장의 길이가 전부 동일하게 된것을 알 수 있습니다. 
    # 기계는 이제 이들을 하나의 행렬로 보고, 병렬 처리를 할 수 있습니다. 
    # 또한, 0번 단어는 사실 아무런 의미도 없는 단어이기 때문에 자연어 처리하는 과정에서 기계는 0번 단어를 무시하게 될 것입니다.
    # 이와 같이 데이터에 특정 값을 채워서 데이터의 크기(shape)를 조정하는 것을 패딩(padding)이라고 합니다. 
    # 숫자 0을 사용하고 있다면 제로 패딩(zero padding)이라고 합니다.


    return padded_np
# padded_np = integer_encoding(word_to_index, raw_data)
# print(padded_np)






# with open('output.csv', 'w') as output_csv:
#   fields = ['songid', '형태소', '품사', '명사']
#   output_writer = csv.DictWriter(output_csv, fieldnames=fields)
 
#   output_writer.writeheader()
#   for item in big_list:
#     output_writer.writerow(item)

 
        
# noise Removal
