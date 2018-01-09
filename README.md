# Quora Question Pairs

On Q&A-Portals like Quora, Yahoo! Answers, Ask.fm, WikiAnswers or StackExchange were asked und answered tausands of questions every day. 6 Million of the over 13 million questions on Quora were asked only in the last year (march 2017). Thats are more than 16,000 new questions per day. For this amount of questions it is likely that someone want to ask an question which is alreecause the portal could list equal question to the user at the point he formulates his question. Maybe someone has already answered one of those questions and the questioner can get the answer to his question immediately. Also the seekers of answers find all answers central on one location and they don't have to look on ten versions of the question for an answer. The question responders profit likewise because they have to answer only one time per question. This could result in a better discussion for people with different opinions.

## Approach
To solve this task data I do the following three steps:
1. analyse the question-pairs from quora
2. create an algorithm to idetify same question-meanings
3. evaluate the algorithm

The last two steps I repeated as long as I get an satisfying result. For the evaluation I created a validationset because the evaluation on the testset had some restrictions:
- only 5 submissions each day 
- evaluation only with log-loss
- evaluation only for complete dataset

For those reasons I split up 10 percent of the trainingsset for the validationset. So I was able to:
- create as many evaluations per day
- evaluate with accuracy, precision, recall, f1 and log-loss
- evaluate for the following **scopes**:
	- *all*: complete validationset
	- *0_30*: pairs where both questions have between 0 and 30 chars
	- *30_70*: pairs where both questions have between 30 and 70 chars
	- *70_150*: pairs where both questions have between 70 and 150 chars

## Features
I defined five feature groups:
- **Word Count Features** e.g. length of q1, count of words of q1 without stopwords
- **Levenshtein Feature** which calculates the levenshtein-distance for the pair
- **Shared Words Features** e.g. count of common words or jaccard-distance
- **Tf-Idf Features** which calculates the weights of common words
- **Frequency Features** e.g. count of questionpairs which contains q1

Those five feature groups resulted in 43 Features.

## Parameter Optimization
To find the best combination of features I used random search as parameter optimization method. I've done 170 iterations on which I always used new combinations of these features. Then I calculated the scores of the evaluation metrics for the validationset for all scopes and saved the results in an CSV-file. The next iteration began. Here you can see how the CSV-file looked like:

[[/screenshots/csv.png|alt=CSV-file example]]

For modell training I choosed XGBoost and the parameters of XGBoost I also varied, for example:
- max_depth: 6 / 7 / 8
- num_boosting_rounds: 2000 / 2500

The preprocessing of the questions I implemented with a Pipeline and FeatureUnion. I tryed the following:
- convert all words to lower
- use TokTokTokenizer
- use WordNetLemmatizer

After the training I rounded the resulting predictions to 0 or 1 to make the evaluation possible. Therefor I tryed rounding on 0.4, 0.5 and 0.6 to find the best rounding boundary for the highest f1.

## Results
I achieved an accuracy of over 87 percent for the validationset. The best feature was the Levenshtein-Feature followed from the Tf-Idf Features. I discovered that with only a low amount of features a good accuracy could be achieved.
Also the usage of a tokenizer and lemmatizer increased the accuracy and rounding at 0.4 resulted in the highest f1. Converting all questions to lowercase also add up to a better accuracy.

## Code
For install instructions go [here](https://github.com/stefantaubert/quora-competition/tree/master/src).
If you want to check out me source code you can look up at my GitHub account here: https://github.com/stefantaubert/quora-competition
