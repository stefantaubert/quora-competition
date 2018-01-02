# Code for Quora Competition on Kaggle
My code for quora-question-pairs.

# installation
only support for python 3.6.0

## get data
Traindata:
https://www.kaggle.com/c/quora-question-pairs/download/train.csv.zip

Testdata:
https://www.kaggle.com/c/quora-question-pairs/download/test.csv.zip

save for example to "/data"

## config
create folder "config" in dir
create file "data_root" in dir "config"
insert your root path to your data files in the first line of "data_root"
- e.g. D:/datasets (win) or /datasets (unix)
insert path for stanford-postagger in the second line of "data_root"
- e.g. C:/stanford-postagger-full-2017-06-09/stanford-postagger.jar (win) or /code/libs/stanford-postagger-full-2017-06-09/stanford-postagger.jar (unix)
insert path for java in the third line of "data_root"
- e.g. C:/Program Files/Java/jdk1.8.0_101 (win) or /usr/lib/jvm/java-8-openjdk-amd64/bin/java (unix)

## folders
create folders "features", "submissions", "evaluation" and "evaluation/runs" in your data directory

## python libraries
pip install pandas sklearn matplotlib fuzzywuzzy nltk numpy editdistance python-Levenshtein 

## xgboost install for gpu-support

### windows 
http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/

### linux
https://github.com/dmlc/xgboost/blob/master/doc/build.md

- if you have old version of cmake: deinstall old version and install new version from https://cmake.org/download/ to usr/local/bin

- open terminal in src/main
	git clone --recursive https://github.com/dmlc/xgboost
	cd xgboost; make -j4
	cd build
	/usr/local/bin/cmake .. -DUSE_CUDA=ON
	make -j
	cd ..
	cd python-package; sudo python setup.py install