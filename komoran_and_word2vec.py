from konlpy.tag import Komoran

# from gensim.model import Word2Vec
import time
import csv
import pandas as pd

# 20210529 품사활용 + 정규식
from konlpy.tag import Mecab
import konlpy
import nltk
from chunk_counter import chunk_counter
from analizing_words import get_stopwords, indexing
import re
import numpy as np

komoran = Komoran()
mecab = Mecab()

with open('./first_project/Crawled_Updated.csv') as input_csv:
    reader = csv.DictReader(input_csv, delimiter=',')
    result_list = []
    for row in reader:
        result_list.append(row)
    crawled = pd.DataFrame(result_list)
# print(crawled)


# crawled = pd.read_csv(r"C:\Users\sgi40\OneDrive\WallPaper\github\MultiCamNLP\CrawlingPJ\csv\Crawled_Updated.csv")
# lyrics_corpus = crawled['lyrics'].tolist()
# l = []
# for i in lyrics_corpus:
#     l.append(i.replace("\r\n"," "))

# # 형태소 + 구문제한도 가능한지 확인 필요
# necessaries = []
# for lyric in lyrics_corpus:
#     each_n = []
    
#     # pos_lyrics = komoran.pos(lyric)
#     pos_lyrics = mecab.pos(lyric)
#     # 사용자 단어사전
#     for hts in pos_lyrics:
#         if hts[1] in ['SL', 'NNG', 'VV', 'NR', 'NP', 'VA', 'MAG', 'XR']:
#             # each_n.append(hts[0])
#             #  20210530
#             each_n.append(hts)
#     necessaries.append(each_n)

#     # 20210529
#     # https://konlpy.org/en/latest/examples/chunking/
#     # Define a chunk grammar, or chunking rules, then chunk
#     grammar = """
#     NP: {<N.*>+<Suffix>?}   # Noun phrase
#     VP: {<V.*>*}            # Verb phrase
#     AP: {<A.*>*}            # Adjective phrase
#     """
#     parser = nltk.RegexpParser(grammar)
    
#     # Chunk filtering lets you define what parts of speech you do not want in a chunk and remove them.
#     # 명사구 외의 형식의 구를 제외함으로서 명사구를 남기는 방식
#     chunk_grammar = """NP: {<.*>+}
#                        }<VB.?|IN>+{"""
#     # The inverted brackets }{ indicate which parts of speech you want to filter from the chunk. <VB.?|IN>+ will filter out any verbs or prepositions
#     chunks = parser.parse(pos_lyrics)
#     # print("# Print whole tree")
#     # print(chunks.pprint())
#     chunked = list()
#     # print("\n# Print phrases only")
#     for subtree in chunks.subtrees():
#         if subtree.label()=='NP':#명사구
#         # if subtree.label()=='VP':#동사구
#         # if subtree.label()=='AP':#형용사구
#             # print(list(subtree))
#             # print(' '.join((e[0] for e in list(subtree))))
#             # print(subtree.pprint())
#             chunked.append(subtree)
#             # chunked.append(subtree.pprint())
#     most_common_vp_chunks = chunk_counter(chunked, 'NP')
#     # print(most_common_vp_chunks)
#     # # Display the chunk tree
#     # chunks.draw()

# mecab, Komoran inner 조인후 
# 언어 모델의 평가 PPL
# PPL의 값이 낮다는 것은 테스트 데이터 상에서 높은 정확도를 보인다는 것이지, 
# 사람이 직접 느끼기에 좋은 언어 모델이라는 것을 반드시 의미하진 않는다

# 조인이 되지 않는 곳의 경우  SOYNLP의 응집 확률순으로 정렬해 정제 해보기
# 희소 문제(sparsity problem)

# n-gram은 뒤의 단어 몇 개만 보다 보니 의도하고 싶은 대로 문장을 끝맺음하지 못하는 경우가 생긴다
# n은 최대 5를 넘게 잡아서는 안 된다고 권장

groupedbyartist = crawled.groupby('artist')['lyrics'].apply('\n\n'.join).reset_index()
# crawled = crawled[:200]# 일부사용

def filter_by_pos (lyric):
    pos_lyrics = []
    # 사용자 단어사전
    # komoran.pos(lyric)
    for hts in  mecab.pos(lyric):
        if hts[1] in ['SL', 'NNG', 'VV', 'NR', 'NP', 'VA', 'MAG', 'XR']:
            # each_n.append(hts[0])
            #  20210530
            pos_lyrics.append(hts)
    return pos_lyrics

# crawled['pos_lyrics'] = crawled.lyrics.apply(filter_by_pos)
groupedbyartist['pos_lyrics'] = groupedbyartist.lyrics.apply(filter_by_pos)

# df, raw_data = indexing(necessaries, stop_words_param=stop_words)#문서전체에 대한 횟수
# df_indexed : feature dictionary is a map of each unique word token in the training data to a vector index
# df_indexed, raw_data = indexing(crawled[crawled['pos_lyrics'].notnull()]['pos_lyrics'], stop_words_param=stop_words)#문서전체에 대한 횟수

try:
    # df_indexed = pd.read_csv('indexed_terms.csv')
    groupedbyartist_indexed = pd.read_csv('grouped_terms.csv')
    
    # print(df_indexed.info()) # 단어별, 형태소별, 순위, 등장횟수 출력
except:
    stop_words= get_stopwords()

    # df_indexed, raw_data = indexing(crawled[crawled['pos_lyrics'].notnull()]['pos_lyrics'], stop_words_param=stop_words, feature_num = 300)#문서전체에 대한 횟수

    # df_indexed.to_csv('indexed_terms.csv',index=False)

    groupedbyartist_indexed, raw_data_grouped= indexing(groupedbyartist[groupedbyartist['pos_lyrics'].notnull()]['pos_lyrics'], stop_words_param=stop_words, feature_num = 50)
    groupedbyartist_indexed.to_csv('grouped_terms.csv',index=False)



def indexing_pos (pos):
    idx_list = []
    for tup in pos:
        idx = None
        for val in df_indexed.values:
            tupl = val[4]
            if isinstance(tupl, str):
                tupl = re.findall('\w+',tupl)
            if all([a==b for a,b in zip(tup, tupl)]):
                idx = val[2]
                break
        if idx is None:# 해당되는 형태소가 없는경우 
            idx = df_indexed.values[-1][2]
        idx_list.append(idx)

    # 사용자 단어사전
    return idx_list
# crawled['indexed_pos'] = crawled.pos_lyrics.apply(indexing_pos)


def indexing_pos_group (pos):
    idx_list = []
    for tup in pos:
        idx = None
        for val in groupedbyartist_indexed.values:
            tupl = val[4]
            if isinstance(tupl, str):
                tupl = re.findall('\w+',tupl)
            if all([a==b for a,b in zip(tup, tupl)]):
                idx = val[2]
                break
        if idx is None:# 해당되는 형태소가 없는경우 
            idx = groupedbyartist_indexed.values[-1][2]
        idx_list.append(idx)

    # 사용자 단어사전
    return idx_list


groupedbyartist['indexed_pos'] = groupedbyartist.pos_lyrics.apply(indexing_pos_group)
print(groupedbyartist)
# print(raw_data)
# This technique of grouping words by their part-of-speech tag is called chunking.
# With chunking in nltk, you can define a pattern of parts-of-speech tags using a modified notation of regular expressions. 
# You can then find non-overlapping matches, or chunks of words, in the part-of-speech tagged sentences of a text.
# chunk_grammar = "AN: {<JJ><NN>}"
# The chunk grammar above will thus match any adjective that is followed by a noun.
# ex. 형용사와 문자가 엮여 있는 구를 해석 단위로 하는 기법
# One such type of chunking is NP-chunking, or noun phrase chunking. A noun phrase is a phrase that contains a noun and operates, as a unit, as a noun.
# 명사구
# chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"
# https://www.codecademy.com/paths/natural-language-processing/tracks/nlp-language-parsing/modules/nlp-language-parsing/lessons/nlp-regex-parsing-intro/exercises/intro-chunking

# DTM 내에 있는 각 단어에 대한 중요도를 계산할 수 있는 TF-IDF 가중치    
# TF-IDF를 사용하면, 기존의 DTM을 사용하는 것보다 보다 더 많은 정보를 고려하여 문서들을 비교할 수 있습니다. 
# (주의할 점은 TF-IDF가 DTM보다 항상 성능이 뛰어나진 않습니다.)
# tf-idf = 특정문서내 문자의 등장횟수에 비례, 특정단어가 등장한 문서의 수에 반비례
# TF-IDF는 모든 문서에서 자주 등장하는 단어는 중요도가 낮다고 판단하며
# , 특정 문서에서만 자주 등장하는 단어는 중요도가 높다고 판단합니다. 
# TF-IDF 값이 낮으면 중요도가 낮은 것이며, TF-IDF 값이 크면 중요도가 큰 것입니다. 



def tf(txt, doc):#tf(d,t) : 특정 문서 d에서의 특정 단어 t의 등장 횟수.
    return doc.count(txt)

def idf(txt):# 특정단어가 등장하는 문서의 갯수
    doc_cnt = 0
    # for doc in crawled.pos_lyrics:
    for doc in crawled.indexed_pos:
        doc_cnt += txt in doc
    return (np.log(crawled.shape[0]/(doc_cnt + 1))+1)

def tfidf(txt, doc):
    return tf(txt,doc)* idf(txt)

def get_tfodf (doc):
    result = []
    for txt in df_indexed[df_indexed.columns.values[2]]:
        result.append(tfidf(txt, doc))
    return result
# crawled['tfidf'] = crawled.indexed_pos.apply(get_tfodf)

def tf_g(txt, doc):#tf(d,t) : 특정 문서 d에서의 특정 단어 t의 등장 횟수.
    return doc.count(txt)

def idf_g(txt):# 특정단어가 등장하는 문서의 갯수
    doc_cnt = 0
    # for doc in crawled.pos_lyrics:
    for doc in groupedbyartist.indexed_pos:
        doc_cnt += txt in doc
    return (np.log(groupedbyartist.shape[0]/(doc_cnt + 1))+1)

def tfidf_g(txt, doc):
    return tf_g(txt,doc)* idf_g(txt)

def get_tfodf_g (doc):
    result = []
    for txt in groupedbyartist_indexed[groupedbyartist_indexed.columns.values[2]]:
        result.append(tfidf_g(txt, doc))
    return result

groupedbyartist['tfidf']= groupedbyartist.indexed_pos.apply(get_tfodf_g)
# 중요도순 + 상하위 10개의 요소 + 각 노래별
groupedbyartist.to_csv('grouped_terms_by_artist.csv',index=False)

print(type(groupedbyartist))
print(groupedbyartist)

def gets_significant_terms_on_each_song(terms,terms_indexed):
    significant_terms_on_each_song = []
    for i in terms.index:
        row = terms.iloc[i]
        zipped = zip(terms_indexed[terms_indexed.columns.values[2]], row.tfidf)
        list_zipped = [[a,b]  for a,b in zipped if b >0]
        df_zipped = pd.DataFrame(list_zipped, columns=['idx_term','tfidf']).sort_values(by='tfidf', ascending=False)
        df_merged = pd.merge(left=df_zipped, right=terms_indexed, left_on='idx_term', right_on=terms_indexed.columns.values[2])
        df_merged_partial = df_merged[['tfidf',df_merged.columns.values[-1]]]
        terms_by_artist = {'artist':row.artist}
        for i in df_merged_partial.index: 
            if i > 50:# 고려할 속성수
                break
            inner_row = df_merged_partial.iloc[i]
            terms_by_artist.update({i:inner_row.values})
        # significant_terms = df_merged_partial.iloc[:num_features]
        # not_significant_terms = df_merged_partial.iloc[-num_features:]
        # significant_terms_on_each_song.append({row.artist:{'significant_terms':significant_terms,'not_significant_terms':not_significant_terms}})
        significant_terms_on_each_song.append(terms_by_artist)
    return pd.DataFrame(significant_terms_on_each_song)
test_df = gets_significant_terms_on_each_song(groupedbyartist, groupedbyartist_indexed)
test_df.to_csv('test_df.csv')
print(test_df)
# for song in crawled.values:
#     # 특정단어의 범위: 전체문서 등장횟수기준 400개의 형태소
#     # tf(d,t) : 특정 문서 d에서의 특정 단어 t의 등장 횟수.
#     # print([(t, lyric.count(t)) for t in df[4]])# 특정단어 정보 포함
#     # https://wikidocs.net/31698
#     # print([tfidf(t, lyric) for t in df[4]])#각문서별 형태소의 중요도를 tf-idf 방식으로 계산
#     # doc_matrix.append([tfidf(t, lyric) for t in df[4]])
#     x = datetime.now()
#     doc_matrix.append({song[2] :[tfidf(t, song[5]) for t in df[4]]})# 6.5~7 sec
#     if len(doc_matrix) >100:# 900개를 전부 계산하니 시간이 너무 걸려서 10개로 추림
#         break
#     print(datetime.now()-x)
# similarity 함수
def cos_sim(A, B):
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))


def get_similarity_matrix(df):
    similarity_mat = []
    for idx1 in df.index:
        temp = []
        similarity_mat.append(temp)
        for idx2 in df.index:
            temp.append(cos_sim(df.iloc[idx1]['tfidf'], df.iloc[idx2]['tfidf']))
    return np.array(similarity_mat)

# 가사간 유사성 출력
# test_mat = get_similarity_matrix(crawled)
# print(test_mat)# similarity 함수

# 가수간 유사성 # tf_idf를 가수별 기준으로 수정해야 함
# 일정 단어 혹은 문장을 기준으로 기존 data와의 유사성 확인후
# 유사성에 따라 정렬

# 주어진 가사에 대한 유사성 출력
def get_similarity(df, string_input):
    string_w_pos = filter_by_pos(string_input)
    pos_w_idx = indexing_pos(string_w_pos)
    string_tfidf = get_tfodf(pos_w_idx)
    similarity_mat = []
    
    for idx1 in df.index:
        row = df.iloc[idx1]
        similarity_mat.append({'songid':row['songid'],'tfidf':cos_sim(row['tfidf'], string_tfidf)})

    return pd.DataFrame(data = similarity_mat)
text_for_testing = '''
Doc, I'm feeling bad
미열이 흐르고 또 어질어질해
There's too much pain
식은땀이 흘러 온몸이 끈끈해
'''
# df_similar = get_similarity(crawled[['songid','tfidf']], text_for_testing)
# df_song_and_artist = crawled[['songid','songname','artist']]
# result = pd.merge(df_similar, df_song_and_artist, left_on='songid', right_on='songid').reindex(df_similar.index)
# print(result.sort_values(by=['tfidf'], ascending=False))




# 단어의 표현 방법은 크게 국소 표현(Local Representation) 방법과 
# 분산 표현(Distributed Representation) 방법으로 나뉩니다. 
# 국소 표현 방법은 해당 단어 그 자체만 보고, 특정값을 맵핑하여 단어를 표현하는 방법이며, 
# 분산 표현 방법은 그 단어를 표현하고자 주변을 참고하여 단어를 표현하는 방법입니다.