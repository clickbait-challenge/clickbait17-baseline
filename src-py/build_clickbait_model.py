#!/usr/bin/python3
""" Build feature schema, train model and safe both to disk"""
import numpy as np
from features import feature as ft
from features.feature_builder import FeatureBuilder
from features.ml import ClickbaitModel
from features.dataset import ClickbaitDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from copy import deepcopy
import scipy.sparse
import sys
import json
import pickle
import os


def usage():
    print(""" Usage:
python3 build_clickbait_model.py <path-to-training-dataset>
For the Clickbait Challenge 2017, the training dataset was clickbait17-validation-170630
""")


def build_new_features(cbd):
    print('defining features')
    common_phrases = ft.ContainsWordsFeature("wordlists/TerrierStopWordList.txt", ratio=True)
    char_3grams = ft.NGramFeature(TfidfVectorizer, o=3, analyzer='char', fit_data=cbd.get_x('postText'), cutoff=3)
    word_3grams = ft.NGramFeature(TfidfVectorizer, o=3, fit_data=cbd.get_x('postText'), cutoff=3)

    stop_word_ratio = ft.ContainsWordsFeature("wordlists/TerrierStopWordList.txt", ratio=True)
    easy_words_ratio = ft.ContainsWordsFeature("wordlists/DaleChallEasyWordList.txt", ratio=True)
    mentions_count = ft.ContainsWordsFeature(['@'], only_words=False)
    hashtags_count = ft.ContainsWordsFeature(['#'], only_words=False)
    clickbait_phrases_count = ft.ContainsWordsFeature("wordlists/DownworthyCommonClickbaitPhrases.txt",
                                                      only_words=False)
    flesch_kincait_score = ft.FleschKincaidScore()
    has_abbrev = ft.ContainsWordsFeature("wordlists/OxfortAbbreviationsList.txt", only_words=False, binary=True)
    number_of_dots = ft.ContainsWordsFeature(['.'], only_words=False)
    start_with_number = ft.StartsWithNumber()
    longest_word_length = ft.LongestWordLength()
    mean_word_length = ft.MeanWordLength()
    char_sum = ft.CharacterSum()
    has_media_attached = ft.HasMediaAttached()
    part_of_day = ft.PartOfDay()
    sentiment_polarity = ft.SentimentPolarity()

    f_builder = FeatureBuilder((char_3grams, 'postText'),
                               (word_3grams, 'postText'),
                               (hashtags_count, 'postText'),
                               (mentions_count, 'postText'),
                               (sentiment_polarity, 'postText'),
                               (flesch_kincait_score, 'postText'),
                               (has_abbrev, 'postText'),
                               (number_of_dots, 'postText'),
                               (start_with_number, 'postText'),
                               (longest_word_length, 'postText'),
                               (mean_word_length, 'postText'),
                               (char_sum, 'postText'),
                               (has_media_attached, 'postMedia'),
                               (part_of_day, 'postTimestamp'),
                               (easy_words_ratio, 'postText'),
                               (stop_word_ratio, 'postText'),
                               (clickbait_phrases_count, 'postText'))

    for file_name in os.listdir("wordlists/general-inquirer"):
        f = ft.ContainsWordsFeature("wordlists/general-inquirer/" + file_name)
        f_builder.add_feature(feature=f, data_field_name='postText')

    char_3grams_mc = ft.NGramFeature(TfidfVectorizer, o=3, analyzer='char', fit_data=cbd.get_x('targetParagraphs'), cutoff=3)
    word_3grams_mc = ft.NGramFeature(TfidfVectorizer, o=3, fit_data=cbd.get_x('targetParagraphs'), cutoff=3)

    f_builder.add_feature(feature=char_3grams_mc, data_field_name='targetParagraphs')
    f_builder.add_feature(feature=word_3grams_mc, data_field_name='targetParagraphs')
    f_builder.add_feature(feature=flesch_kincait_score, data_field_name='targetParagraphs')
    f_builder.add_feature(feature=mean_word_length, data_field_name='targetParagraphs')

    print('building features')
    f_builder.build(cbd)
    print('storing feature schema as feature_builder.pkl')
    pickle.dump(obj=f_builder, file=open("feature_builder.pkl", "wb"))
    return f_builder


if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()
        exit(1)
    if os.path.exists(sys.argv[1]):
        input_dir = os.path.abspath(sys.argv[1])
    else:
        print("invalid path for input directory: " + str(sys.argv[1]))
        exit(1)
    cbd = ClickbaitDataset("{}/instances.jsonl".format(input_dir),
                           "{}/truth.jsonl".format(input_dir))

    f_builder = build_new_features(cbd)
    x = f_builder.build_features

    print('training model')
    cbm = ClickbaitModel()
    cbm.regress(x, cbd.get_y(), Ridge(alpha=3.5), evaluate=False)
    print('stroring trained model as model_trained.pkl')
    cbm.save("model_trained.pkl")
