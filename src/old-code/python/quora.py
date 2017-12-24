

# In[29]:

#LogLoss für Ähnlichkeit der Fragen
def comparison(frage1, frage2, is_duplicate):
    equal_score = get_question_equal_score(frage1, frage2)
    equal_terms_count = equal_score[0]
    term_count = equal_score[1]    
    return equal_terms_count / term_count

get_matching_questions(comparison)

# In[21]:

qid_dict = {}

for i,series in df_train.iterrows():
    if series['qid1'] not in qid_dict:
        qid_dict[series['qid1']] = series['question1']
    if series['qid2'] not in qid_dict:
        qid_dict[series['qid2']] = series['question2']


# In[22]:

for k in qid_dict:
    print(qid_dict[k], clean_string(qid_dict[k]))
    #qid_dict[k] = clean_string(qid_dict[k])


# In[33]:

#LogLoss für Ähnlichkeit der Fragen
def comparison(frage1, frage2, is_duplicate):
    q1_clean = clean_string(frage1)
    q2_clean = clean_string(frage2)
    equal_score = get_question_equal_score(q1_clean, q2_clean)
    equal_terms_count = equal_score[0]
    term_count = equal_score[1]
    return equal_terms_count / term_count

get_matching_questions(comparison)


# In[58]:

def get_question_equal_score_core(smaller_parts, larger_parts):
    equal_terms_count = 0
    term_count = 0
    term_count = len(smaller_parts)
    for str in smaller_parts:
        if str in larger_parts:
            equal_terms_count = equal_terms_count + 1
                    
    return [equal_terms_count, term_count]

def get_question_equal_score_ext(question1, question2):
    q1_parts = question1.lower().split()
    q2_parts = question2.lower().split()
    
    for str in q1_parts:
        q1_parts.remove("?")
        q1_parts.remove(".")
        q1_parts.remove("!")
        q1_parts.remove(",")
        q1_parts.remove(":")
        q1_parts.remove("\"")        
        q1_parts.remove("(")
        q1_parts.remove(")")
        
            
    for str in q2_parts:
        q2_parts.remove("?")
        q2_parts.remove(".")
        q2_parts.remove("!")
        q2_parts.remove(",")
        q2_parts.remove(":")
        q2_parts.remove("\"")        
        q2_parts.remove("(")
        q2_parts.remove(")")
            
    if len(q1_parts) <= len(q2_parts):
        return get_question_equal_score_core(q1_parts, q2_parts)
    else:
        return get_question_equal_score_core(q2_parts, q1_parts)


# In[54]:

#Analyse von clean_string
def comparison(frage1, frage2, is_duplicate):
    origial_equal_score = get_question_equal_score_ext(frage1, frage2, 2)
    q1_clean = clean_string(frage1)
    q2_clean = clean_string(frage2)
    equal_score = get_question_equal_score_ext(q1_clean, q2_clean, 2)
    equal_terms_count = equal_score[0]
    term_count = equal_score[1]
    
    original_score = origial_equal_score[0] / origial_equal_score[1]
    score = equal_terms_count / term_count
    
    if (is_duplicate == 0):
        if (original_score < score):
            print("Dup: {}\nFrage1: {}\nFrage2: {}\nF1_clean: {}\nF2_clean: {}".format(is_duplicate, frage1,frage2,q1_clean,q2_clean))
    else:
        if (original_score > score):
             print("Dup: {}\nFrage1: {}\nFrage2: {}\nF1_clean: {}\nF2_clean: {}".format(is_duplicate, frage1,frage2,q1_clean,q2_clean))
 
    return equal_terms_count / term_count

get_matching_questions(comparison)

#LogLoss für Ähnlichkeit der Fragen
def comparison(frage1, frage2, is_duplicate):
    q1_clean = clean_string(frage1)
    q2_clean = clean_string(frage2)
    equal_score = get_question_equal_score_ext(q1_clean, q2_clean)
    equal_terms_count = equal_score[0]
    term_count = equal_score[1]
    if term_count == 0:
        return 0
    else:
        return equal_terms_count / term_count

get_matching_questions(comparison)