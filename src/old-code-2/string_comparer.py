from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))

def word_match_share(question1, question2):
    q1words = {}
    q2words = {}
    for word in str(question1).lower().split():
        q1words[word] = 1
    for word in str(question2).lower().split():
        q2words[word] = 1
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def word_match_share_without_stopwords(question1, question2):
    q1words = {}
    q2words = {}
    for word in str(question1).lower().split():
        if word not in eng_stopwords:
            q1words[word] = 1
    for word in str(question2).lower().split():
        if word not in eng_stopwords:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def get_question_equal_score_core(smaller_parts, larger_parts):
    equal_terms_count = 0
    term_count = 0
    term_count = len(smaller_parts)
    for str in smaller_parts:
        if str in larger_parts:
            equal_terms_count = equal_terms_count + 1

    return [equal_terms_count, term_count]

def get_question_equal_score(question1, question2):
    q1_parts = question1.lower().split()
    q2_parts = question2.lower().split()

    if len(q1_parts) <= len(q2_parts):
        return get_question_equal_score_core(q1_parts, q2_parts)
    else:
        return get_question_equal_score_core(q2_parts, q1_parts)
