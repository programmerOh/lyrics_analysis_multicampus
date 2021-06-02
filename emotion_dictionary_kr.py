import requests
import certifi
import socket
import ssl
import pandas as pd
from bs4 import BeautifulSoup
import re
from konlpy.tag import Mecab

mecab = Mecab()
# print(mecab.morphs('영등포구청역에 있는 맛집 좀 알려주세요.'))
cert = certifi.where()
print(cert)
def get_stop_words():
    url = 'https://www.ranks.nl/stopwords/korean'
    # url = 'https://www.naver.com/'

    context = ssl.create_default_context()
    conn = context.wrap_socket(socket.socket(socket.AF_INET),server_hostname="www.ranks.nl")
    conn.connect(("www.ranks.nl", 443))
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
        , 'Accept-Encoding': 'gzip, deflate, br'
        , 'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6'
        , 'Cache-Control': 'max-age=0'
        , 'Connection': 'keep-alive'
        , 'Cookie': 'wgSession=jFsQ2b6HQXDOglW8ocB0Jw; __utmc=73930001; __utmz=73930001.1622114508.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); _ga=GA1.2.1150684315.1622114508; _gid=GA1.2.1028009705.1622117227; __utma=73930001.1150684315.1622114508.1622117047.1622149127.3; __utmt=1; __utmb=73930001.1.10.1622149127'
        , 'Host': 'www.ranks.nl'
        , 'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="90", "Google Chrome";v="90"'
        , 'sec-ch-ua-mobile': '?0'
        , 'Sec-Fetch-Dest': 'document'
        , 'Sec-Fetch-Mode': 'navigate'
        , 'Sec-Fetch-Site': 'none'
        , 'Sec-Fetch-User': '?1'
        , 'Upgrade-Insecure-Requests': '1'
        , 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
    }
    res = requests.get(url=url,headers=headers, timeout=5)
    soup = BeautifulSoup(res.text, 'html.parser')
    tds = soup.select('div.panel-body > table > tbody > tr > td')
    for td in tds:
        pass
# get_stop_words()

def load_dictionary_on_emotion():

    url = 'http://dilab.kunsan.ac.kr/knu/knu.html'
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    elements = soup.select('body > input')
    param_for_query=[]
    for el in elements:
        param_for_query.append(el.attrs['value'].strip())

    words_for_sub = []
    df_emotion= pd.DataFrame()
    for param in param_for_query:
        url = f'http://dilab.kunsan.ac.kr/knu/text/{param}.txt'
        res = requests.get(url)
        list = res.text.strip().split('\r\n')
        dictionary_emotion = {'단어':[],'어근':[],'감성':[]}
        for el in list:
            inner_list = el.strip().split('\n')
            for inner_el in inner_list:
                words = inner_el.split(':')
                if dictionary_emotion.get(words[0]) is not None:
                    dictionary_emotion.get(words[0]).append(words[1].strip())
                else:
                    words_for_sub.append(inner_el)
        tuples = [(param, i) for i in range(len(dictionary_emotion['단어']))]
        index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = pd.DataFrame(dictionary_emotion, index=index)
        df_emotion = pd.concat([df_emotion, df])
    return df_emotion

# def temp_request(url, df):
#     res = requests.get(url)
#     soup = BeautifulSoup(res.text, 'html.parser')
#     elements = soup.select('div.single-post-area.style-2 a.post-title.mb-2')

#     for i, element in enumerate(elements):
#         child_url = element.attrs['href']
#         print(str(i+1), child_url, element.text)

#         res_child = requests.get(f'https://www.lyrics.co.kr{child_url}')
#         soup_child = BeautifulSoup(res_child.text, 'html.parser')
#         elements_child = soup_child.select('div.blog-content>p>div')
#         for element_child in elements_child:
#             print(element_child.text)
#             string=''
#             for text in element_child.contents:
#                 if isinstance(text, str):
#                     tokens = text.split()# tokenization
#                     for token in tokens:
#                         if (df[df['단어']==token].size>0) :#or (df[df['어근']==token].size>0):
#                             print(df[df['단어']==token])
#                             # print(df[df['어근']==token])
#                             print(df[df['단어']==token].shape)
#                             # print(df[df['어근']==token].size)
#                             print(df[df['어근']==token]['감성'])

#                         # print(re.search('[ㄱ]'))
#                     string += f'\n{text}'
#             print(string)

# df = load_dictionary_on_emotion()
# temp_request('https://www.lyrics.co.kr/#gsc.tab=0', df)