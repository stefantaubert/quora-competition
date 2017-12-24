Python Version: 3.6.0 (höher geht nicht)
Bibliotheken: pip install pandas sklearn matplotlib fuzzywuzzy nltk numpy editdistance python-Levenshtein 

xgboost install instruction:
Quelle: https://github.com/dmlc/xgboost/blob/master/doc/build.md

Linux:
Terminal im Projektverzeichnis öffnen

	git clone --recursive https://github.com/dmlc/xgboost
	cd xgboost; make -j4

cuda muss installiert sein

	cd build; /usr/local/bin/cmake .. -DUSE_CUDA=ON
	make -j

- wenn cmake veraltet ist muss die alte version gelöscht werden und neu von https://cmake.org/download/ installiert werden, dabei wird aber in usr/local/bin installiert, anstatt zu usr/bin, deswegen muss
	/usr/local/bin/cmake .. -DUSE_CUDA=ON 
ausgeführt werden
https://geeksww.com/tutorials/operating_systems/linux/installation/downloading_compiling_and_installing_cmake_on_linux.php


	cd ..; cd python-package; sudo /code/libs/anaconda3/envs/py361/bin/python3.6 setup.py install

falls python nicht richtig gelinkt ist, dann muss der exakte pfad zur pythonexecutable angegeben werden anstatt python

/code/libs/anaconda3/envs/py361/bin/python3.6 /code/sttau/quora-bachelor-thesis/python/script_evaluation.py /datasets/sttau/
