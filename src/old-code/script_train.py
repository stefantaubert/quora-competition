import sys
from time import clock

import ModelTraining
import postprocessing

import prediction

start = clock()
root = sys.argv[1]
ModelTraining.train_and_save_model(root)
prediction.predict_and_write_testdata(root)
postprocessing.execute(root)

print('Overall duration: ', round(clock()-start, 2), 'seconds')
