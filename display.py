import tkinter as tk
import twitterSentimentAnalysis
from helper import helperClass

classifier = twitterSentimentAnalysis.get_classifier()
helper = helperClass()

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 400,  relief = 'raised')
canvas1.pack()

lbl1 = tk.Label(root, text='Tweet Sentiment Analysis')
lbl1.config(font=('verdana', 14))
canvas1.create_window(200, 25, window=lbl1)

lbl2 = tk.Label(root, text='Type your Tweet:')
lbl2.config(font=('verdana', 10))
canvas1.create_window(200, 75, window=lbl2)

entry1 = tk.Entry(root, width=50) 
canvas1.create_window(200, 100, window=entry1)

def entered_tweet_feature_set():
    x1 = entry1.get()
    custom_tweet_feature_set = helper.bag_of_words(x1)
    return custom_tweet_feature_set

def get_probability_result():
    prob_result = classifier.prob_classify(entered_tweet_feature_set())
    return prob_result

def get_tweet_probability():
    return get_probability_result().max()

def get_tweet_analysis():

    lbl3 = tk.Label(root, text= 'The Probability Result for the entered Tweet is:',font=('verdana', 10))
    canvas1.create_window(200, 200, window=lbl3)
    
    lbl4 = tk.Label(root, text= get_tweet_probability(),font=('verdana', 12, 'bold'))
    canvas1.create_window(200, 220, window=lbl4)

    lbl5 = tk.Label(root, text= 'The Positive Probability is:',font=('verdana', 10))
    canvas1.create_window(200, 240, window=lbl5)

    lbl6 = tk.Label(root, text= str(round((get_probability_result().prob("pos") *100), 2)) + "%" ,font=('verdana', 12, 'bold'))
    canvas1.create_window(200, 260, window=lbl6)

    lbl7 = tk.Label(root, text= 'The Negative Probability is:',font=('verdana', 10))
    canvas1.create_window(200, 280, window=lbl7)

    lbl8 = tk.Label(root, text=  str(round((get_probability_result().prob("neg") *100), 2)) + "%",font=('verdana', 12, 'bold'))
    canvas1.create_window(200, 300, window=lbl8)
    
btn1 = tk.Button(text='Get Tweet Sentiment Analysis', command=get_tweet_analysis, bg='brown', fg='white', font=('verdana', 9, 'bold'))
canvas1.create_window(200, 140, window=btn1)

root.mainloop()