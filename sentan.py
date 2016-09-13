import nltk

pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    tweets.append((words_filtered, sentiment))

test_tweets = [
    (['feel','happy', 'this', 'morning'], 'positive'),
    (['larry', 'friend'], 'positive'),
    (['not', 'like', 'that', 'man'], 'negative'),
    (['house', 'not', 'great'], 'negative'),
    (['your', 'song', 'annoying'], 'negative')]





###########nltk
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features




############### SPARK
from pyspark import SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
sc= SparkContext()



## adapt spark:
tweets_rdd_train = sc.parallelize(tweets,10)
tweets_rdd_test = sc.parallelize(test_tweets,10)
def extract_features_spark(document):
    label = document[1]
    document_words = set(document[0])
    features = []
    for word in word_features:
        features+= [word in document_words]
    if label =='positive':
        lab = 1
    elif label =='negative':
        lab = -1
    return LabeledPoint(lab, features)###############################################!! changer le 1.0
data_train = tweets_rdd_train.map(extract_features_spark)
data_test = tweets_rdd_test.map(extract_features_spark)


model = NaiveBayes.train(data_train, 1.0)
predictionAndLabel = data_test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / data_test.count()
print(predictionAndLabel.collect(),accuracy)



###########nltk
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
tweet = 'Larry is not not my friend'
print classifier.classify(extract_features(tweet.split()))








