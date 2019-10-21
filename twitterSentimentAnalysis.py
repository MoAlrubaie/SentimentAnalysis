from nltk.corpus import twitter_samples
from nltk import classify
from nltk import NaiveBayesClassifier
from random import shuffle
from helper import helperClass

helper = helperClass()

#preview the data fields
print (twitter_samples.fileids())

#Split the data in groups
positive_tweets = twitter_samples.strings('positive_tweets.json') 
negative_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = twitter_samples.strings('tweets.20150430-223406.json')

#View the length of each group
print ("Positive Tweets Length: " + str(len(positive_tweets)))
print ("Negative Tweets Length: " + str(len(negative_tweets)))
print ("All Tweets Length: " + str(len(all_tweets)))

#preview samples from each group of tweets
for strings in (positive_tweets[:5], negative_tweets[:5], all_tweets[:5]):
    print(strings)
    print(4*"------")

print("******************************************")
print("******************************************")

custom_tweet = "RT @Twitter @mohafad87 Hello World! Have a great day all. :) #good #morning http://moalrubaie.com"

#view the clean tweet from the custom
print(helper.clean_tweet(custom_tweet))

#use bag of words method
print (helper.bag_of_words(custom_tweet))

#positive & negative tweets feature set
positive_tweets_feature_set = []
negative_tweets_feature_set = []

for tweet in positive_tweets:
    positive_tweets_feature_set.append((helper.bag_of_words(tweet), 'pos'))    
 
for tweet in negative_tweets:
    negative_tweets_feature_set.append((helper.bag_of_words(tweet), 'neg'))

#shuffle before set test and train data 
shuffle(positive_tweets_feature_set)
shuffle(negative_tweets_feature_set)

test_set = positive_tweets_feature_set[:1000] + negative_tweets_feature_set[:1000]
train_set = positive_tweets_feature_set[1000:] + negative_tweets_feature_set[1000:]

#create the NaiveBayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

#get the accuracy
classifier_accuracy = classify.accuracy(classifier, test_set)
print("The Accuracy is: " + str(round((classifier_accuracy *100), 2)) + "%" )

#show the most informative features 
print (classifier.show_most_informative_features(5))

#Get the classifier to be used in the display class for custom tweets purpose only
def get_classifier():
    return classifier