import pandas as pd
import Paths
import random
import hashlib
import settings

selected_feature_ids = []
seed = settings.seed_features
random.seed(seed)

never_include = [
    #     'intersect',
    #     'q1_freq',
    #     'q2_freq',
    #     'levenshtein',
    #     'jaccard',
    #     'common_words',
    #     'word_match',
    #     'word_match_stops',
    #     'tfidf_wm',
    #     'tfidf_wm_stops',
    #     'len_q1',
    #     'len_q2',
    #     'min_len',
    #     'max_len',
    #     'len_diff',
    #     'len_ratio',
    #     'len_q1_blank',
    #     'len_q2_blank',
    #     'min_len_blank',
    #     'max_len_blank',
    #     'len_diff_blank',
    #     'len_ratio_blank',
    #     'count_word_q1',
    #     'count_word_q2',
    #     'min_word_count',
    #     'max_word_count',
    #     'count_words_diff',
    #     'count_words_ratio',
    #     'total_words',
    #     'min_unique_word_count',
    #     'max_unique_word_count',
    #     'len_words_q1_unique',
    #     'len_words_q2_unique',
    #     'diff_unique_word_count',
    #     'ratio_unique_word_count',
    #     'total_unique_words',
    #     'len_words_q1_unique_no_stops',
    #     'len_words_q2_unique_no_stops',
    #     'min_unique_nostops_word_count',
    #     'max_unique_nostops_word_count',
    #     'count_words_diff_unique_no_stops',
    #     'count_words_ratio_unique_no_stops',
    #     'total_unique_no_stopwords',
    #     'char_diff_unique_no_stops',
    'fuzz_qratio',
    'fuzz_WRatio',
    'fuzz_partial_ratio',
    'fuzz_partial_token_set_ratio',
    'fuzz_partial_token_sort_ratio',
    'fuzz_token_set_ratio',
    'fuzz_token_sort_ratio',
]

# besteht aus der Schnittmenge der Top 3 bzw Top 1 aller Versuche.
always_include = [
    'intersect',
    'q1_freq',
    'q2_freq',
    'levenshtein',
    'word_match_stops',
    'tfidf_wm_stops',
]

always_include = []

never_include.extend(always_include)


def select_features_at(root, iteration):
    '''Hinweis: durch das erneute schreiben geht in machen Fällen die letzte Kommastelle verloren oder ändert sich, sollte aber unbedeutend sein.'''
    Paths.init(root)
    random.seed(seed)
    x_train = pd.read_csv(Paths.Get_TRAIN_FEATURES_BACKUP_Path(), encoding="ISO-8859-1")
    all_features = list(x_train)
    all_include_features = [f for f in all_features if f not in never_include]

    selected_features = []
    feature_id = 0
    selected_feature_ids = []

    if iteration == -1:
        selected_features = always_include
    else:
        for i in range(0, iteration):
            while feature_id == 0 or feature_id in selected_feature_ids:
                count = random.randint(1, len(all_include_features))
                selected_features = random.sample(all_include_features, count)
                selected_features.extend(always_include)
                feature_id = hashlib.md5(("".join(sorted(selected_features))).encode('utf-8')).hexdigest()

            selected_feature_ids.append(feature_id)

    for feature in all_features:
        if feature not in selected_features:
            x_train.drop(feature, axis=1, inplace=True)

    x_train.to_csv(Paths.Get_TRAIN_FEATURES_Path(), index=False)

def select_features(root):
    '''Hinweis: durch das erneute schreiben geht in machen Fällen die letzte Kommastelle verloren oder ändert sich, sollte aber unbedeutend sein.'''
    Paths.init(root)
    x_train = pd.read_csv(Paths.Get_TRAIN_FEATURES_BACKUP_Path(), encoding="ISO-8859-1")
    all_features = list(x_train)
    all_include_features = [f for f in all_features if f not in never_include]

    selected_features = []
    feature_id = 0

    while feature_id == 0 or feature_id in selected_feature_ids:
        count = random.randint(1, len(all_include_features))
        selected_features = random.sample(all_include_features, count)
        selected_features.extend(always_include)
        feature_id = hashlib.md5(("".join(sorted(selected_features))).encode('utf-8')).hexdigest()

    selected_feature_ids.append(feature_id)

    for feature in all_features:
        if feature not in selected_features:
            x_train.drop(feature, axis=1, inplace=True)

    x_train.to_csv(Paths.Get_TRAIN_FEATURES_Path(), index=False)
