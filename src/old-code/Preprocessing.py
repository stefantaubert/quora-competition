import Paths
import pandas as pd
from time import clock




def __process_questions__(data):
    data.fillna('', inplace=True) # "" wird sonst als float interpretiert
    data.question1 = data.question1.map(lambda x: x.lower()) 
    data.question2 = data.question2.map(lambda x: x.lower())
    # hier noch standardisieren, stemmen, lemmatisieren und evtl tokenisieren und mit leerzeichen trennen

def foo():
    pipeline = Pipeline([
        ('fill_na', DFTransform(lambda X: X.fill('NA'))),
        ('q1_q2', DFFeatureUnion([
            ('q1', Pipeline([
                ('select', DFTransform(select_column('question1'))),
                ('lower', DFTransform(lambda X: X.lower()))
            ])),
        ])),
    ])



# 1 = Train, 2 = Test, 3 = Beides, 4 = Tmp
def execute(MODE=1, root='/media/stefan/main-data/my-uni-files/VI/bachelorarbeit/data/'):
    '''Ließt die Daten und bereitet sie für die Feature Extraktion vor. 
    Dazu werden alle anderen Spalten bis auf Q1 und Q2 entfernt jede Frage kleingeschrieben.'''

    start = clock()
    Paths.init(root)

    if MODE == 1 or MODE == 3:
        data = pd.read_csv(Paths.Get_TRAIN_DATA_Path())
        data = data.drop(['id', 'qid1', 'qid2', 'is_duplicate'], axis=1)  # Spalten entfernen
        __process_questions__(data)
        data.to_csv(Paths.Get_TRAIN_PREPROCESSED_Path(), index=False)

    if MODE == 2 or MODE == 3:
        data = pd.read_csv(Paths.get_test_data_path())
        data = data.drop(['test_id'], axis=1) #Spalten entfernen
        __process_questions__(data)
        data.to_csv(Paths.Get_TEST_PREPROCESSED_Path(), index=False)

    if MODE == 4:
        data = pd.read_csv(Paths.Get_TMP_DATA_Path(), encoding="ISO-8859-1")
        __process_questions__(data)
        data.to_csv(Paths.Get_TMP_PREPROCESSED_Path(), index=False)

    print("finished")
    print('Duration: ', round(clock()-start, 2), 'seconds')