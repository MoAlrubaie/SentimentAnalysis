from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import string
import re

happy_emotion_icons_set = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
sad_emotion_icons_set = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
 
all_emotions_icons = happy_emotion_icons_set | sad_emotion_icons_set
class helperClass():

    def __init__(self):
        self.stopwords= stopwords.words('english')
        self.stemmer = PorterStemmer()

    def clean_tweet(self,tweet):
        clean_tweets= []
        #Remove hashtags
        tweet = re.sub(r'#', '', tweet)
        #Remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        #Remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        #remove all single characters
        tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', tweet)

        #Tokenize tweet
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)
 
        for word in tweet_tokens:
            #remove stopwords
            #remove emotion icons
            #remove punctuation
            #stemming word
            if (word not in self.stopwords and word not in all_emotions_icons and word not in string.punctuation): 
                stem_word = self.stemmer.stem(word)
                clean_tweets.append(stem_word)
 
        return clean_tweets
    
    def bag_of_words(self,tweet):
        tweet_words = self.clean_tweet(tweet)
        words_dict = dict([word, True] for word in tweet_words)
        return words_dict
