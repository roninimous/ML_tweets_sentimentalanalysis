from random import random
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
# import stopwords as words
import stopwords as stopwords
from flask import Blueprint, render_template, request, redirect, session
import matplotlib.pyplot as plt
import os
import tweepy
import csv
import re
from textblob import TextBlob
import matplotlib
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import wordninja
# if SpellChecker is not working install pyspellchecker
from spellchecker import SpellChecker
from collections import Counter
import nltk
import math
import random
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.add("amp")
matplotlib.use("agg")

# register this file and directory as a blueprint
second = Blueprint(
    "second", __name__, static_folder="static", template_folder="template"
)


# render page when url is called
@second.route("/sentiment_analyzer")
def sentiment_analyzer():
    if "user_id" in session:
        return render_template("sentiment_analyzer.html")
    else:
        return redirect("/")



# render zeroshot text page when url is called
@second.route("/zeroshot_analyzer")
def zeroshot_analyzer():
    if "user_id" in session:
        return render_template("zeroshot_analyzer.html")
    else:
        return redirect("/")


# class with main logic


class SentimentAnalysis:
    def __init__(self):
        self.tweets = []
        self.tweetText = []
        self.tweetNegative = []
        self.tweetPositive = []
        self.tweetNeutral = []
        self.textArray = []
        self.status = []
        self.polarity = []
        self.subjectivity = []

    # This function is for zeroshot classification
    def ZeroShotClassification(self, text):
        # need to install pytorch or tensorflow > 2.0

        classifier = pipeline("zero-shot-classification")
        # If No model was supplied, defaulted to facebook/bart-large-mnli (
        # https://huggingface.co/facebook/bart-large-mnli)
        sequence = text
        candidate_labels = ["positive", "neutral", "negative"]
        classifier(sequence, candidate_labels)

        test = classifier(sequence, candidate_labels)
        # Get text from dict
        text = test.get("sequence")
        # Get labels
        x = test.get("labels")
        z = test.get("scores")
        # Get values
        label1 = str(x[0])
        label2 = str(x[1])
        label3 = str(x[2])
        if label1 == "positive":
            resultcolor1 = "#51d857"
        elif label1 == "neutral":
            resultcolor1 = "#c2a866"
        elif label1 == "negative":
            resultcolor1 = "#d64d4d"

        if label2 == "positive":
            resultcolor2 = "#51d857"
        elif label2 == "neutral":
            resultcolor2 = "#c2a866"
        elif label2 == "negative":
            resultcolor2 = "#d64d4d"

        if label3 == "positive":
            resultcolor3 = "#51d857"
        elif label3 == "neutral":
            resultcolor3 = "#c2a866"
        elif label3 == "negative":
            resultcolor3 = "#d64d4d"
        # format value with 5 decimals
        score1 = round(z[0], 5)
        score2 = round(z[1], 5)
        score3 = round(z[2], 5)

        return label1, label2, label3, score1, score2, score3, text, resultcolor1, resultcolor2, resultcolor3

    # This function first connects to the Tweepy API using API keys
    def DownloadData(self, keyword, tweets):

        # authenticating
        consumerKey = "omXoE5STLNyhcsSVVcdiB9gcT"
        consumerSecret = "S71nVyTrCQhVwlqrmBknnJrSGRSWogf3uOn9CIyHtooAl0AXpG"
        accessToken = "2347068140-9kJy3imzmvE4Y3DxuzdDBHjLJJh5qPlpufqWhls"
        accessTokenSecret = "Df672DboH25WyxqFQgYjYY0GKXPiu0DNus7c5Kf5yF3lP"
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth, wait_on_rate_limit=True)

        # input for term to be searched and how many tweets to search
        # searchTerm = input("Enter Keyword/Tag to search about: ")
        # NoOfTerms = int(input("Enter how many tweets to search: "))

        tweets = int(tweets)
        # searching for tweets

        # self.tweets = tweepy.Cursor(api.search_tweets, q=f'{keyword} -filter:retweets', lang="en").items(
        #     tweets
        # )
        error_msg=""
        # Link to add search by country: https://twittercommunity.com/t/collect-specific-tweets-from-specific-country-using-tweepy/130655
        try:
            tweet_text_searched = tweepy.Cursor(api.search_tweets, q=f'{keyword} -filter:retweets', lang="en", tweet_mode='extended').items(
                tweets
            )

        # put tweets search into dataframe

            json_data = [r._json for r in tweet_text_searched]
            df = pd.json_normalize(json_data)
        except:
            error_msg="Oops!!! It looks like the APIs or Tokens you are using is invalid!"
            polarity=""
            htmlpolarity=""
            positive=""
            negative=""
            neutral=""
            keyword
            tweets
            self.textArray
            numspam=0
            subjectivity=0
            result_color=""
            wordcloud_error=""

            return (
                polarity,
                htmlpolarity,
                positive,
                negative,
                neutral,
                keyword,
                tweets,
                self.textArray,
                numspam,
                subjectivity,
                result_color,
                wordcloud_error,
                error_msg
            )
        # print(df.full_text[0])

        #---------------wordcloud start
        comment_words = ' '
        stopwords = set(STOPWORDS)



        csvFile = open("result.csv", "a", encoding="utf-8")

        # Use csv writer
        csvWriter = csv.writer(csvFile)

        # creating some variables to store info
        polarity = 0
        subjectivity = 0
        positive = 0
        negative = 0
        neutral = 0
        label = ""
        result_with_text = ""
        numtext = 0
        numspam = 0
        nonreduntweet = 0
        print(df.columns)
        for val in df.full_text:

            # typecaste each val to string
            val = str(val)
            val = self.remove_emojis(val)
            cleanVal = self.clean_tweet_text(self.cleanTweet(val))
            if cleanVal not in self.tweetText:
                self.tweetText.append(cleanVal)

            # split the value
                tokens = cleanVal.split()

                print(cleanVal)

                analysis = TextBlob(cleanVal)
                # add each value of polarity and subjectivity to array then use these array to convert to DF
                self.polarity.append(analysis.sentiment.polarity)
                self.subjectivity.append(analysis.sentiment.subjectivity)


                if analysis.sentiment.polarity == 0:
                    neutral += 1
                    label = "Neutral"
                    self.tweetNeutral.append(cleanVal)
                elif analysis.sentiment.polarity > 0:
                    positive += 1
                    label = "Positive"
                    self.tweetPositive.append(cleanVal)
                elif analysis.sentiment.polarity < 0:
                    negative += 1
                    label = "Negative"
                    self.tweetNegative.append(cleanVal)

                subjectivity += analysis.sentiment.subjectivity
                result_with_text = (
                        "[" + label + "] " + cleanVal
                )

            # check redundancy tweets
                if (result_with_text in self.textArray) == False:
                    self.textArray.append(result_with_text)
                # adding up polarities to find the average later
                    polarity += analysis.sentiment.polarity
                else:
                    numspam = numspam + 1

                print(len(self.textArray))
            else:
                numspam = numspam + 1


        nonreduntweet = tweets-numspam

        # create new dataframe to store important data only
        print(str(len(self.tweetText))+" "+str(len(self.polarity))+" "+str(len(self.subjectivity)))
        df1 = pd.DataFrame({
            'Tweet_texts': self.tweetText,
            'polarity': self.polarity,
            'subjectivity': self.subjectivity
        })
        ## Below code is exactly just like above code but the above ones is much better.
        # df1 = pd.DataFrame(self.tweetText, columns=["Tweet_texts"])
        # df1["polarity"] = self.polarity
        # df1["subjectivity"] = self.subjectivity
        print(df1.columns)

        ##### if analysis is not done in the loop with array, it can be done directly in dataframe by using these 2 lines below.
        # df1['polarity'] = df['full_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        # df1['subjectivity'] = df['full_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        ##### end of analysis

        # polarity values ranging from -1 to 1 are really useful for sentiment analysis
        # but let's convert our data to 3 classes (negative, neutral, and positive) so that we can visualize it
        criteria = [df1['polarity'].between(-1, -0.01), df1['polarity'].between(-0.01, 0.01), df1['polarity'].between(0.01, 1)]
        values = ['negative', 'neutral', 'positive']
        df1['sentiment'] = np.select(criteria, values, 0)
        wordcloud_df = df1
        wordcloud_df['words'] = wordcloud_df.Tweet_texts.apply(lambda x:re.findall(r'\w+', x))
        wordcloud_error=""
        try:
            self.get_smart_clouds(wordcloud_df).savefig("static/images/sentiment_wordclouds.png", bbox_inches="tight")
        except:
            wordcloud_error = "Could not generate word cloud as there is not enough words.;Previous word cloud will be used to display."



        print(df1.Tweet_texts.head(10))


        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        for words in tokens:
            comment_words = comment_words + words + ' '

        # Write to csv and close csv file
        csvWriter.writerow(self.tweetText)
        csvFile.close()

        # calculate percentage of each category
        positive = self.percentage(positive, nonreduntweet)
        negative = self.percentage(negative, nonreduntweet)
        neutral = self.percentage(neutral, nonreduntweet)

        print("tweet:"+str(tweets))
        print("spam:"+str(numspam))
        print("nonredun:"+str(nonreduntweet))
        print(positive)
        print(negative)
        print(neutral)

        # finding average reaction
        polarity = round((polarity / (nonreduntweet)),3)
        subjectivity = round((subjectivity / (nonreduntweet)), 3)


        if polarity == 0:
            htmlpolarity = "NEUTRAL"
            result_color = "#c2a866"
        elif polarity > 0:
            htmlpolarity = "POSITIVE"
            result_color = "#51d857"
        elif polarity < 0:
            htmlpolarity = "NEGATIVE"
            result_color = "#d64d4d"

            # Draw Pie chart
        self.plotPieChart(positive, negative, neutral)
        print(self.tweets)
        print(polarity, htmlpolarity)
        return (
            polarity,
            htmlpolarity,
            positive,
            negative,
            neutral,
            keyword,
            tweets,
            self.textArray,
            numspam,
            subjectivity,
            result_color,
            wordcloud_error,
            error_msg
        )

            #-------------------------------------------------------------------------------------------





        #---------------------------wordcloud end
        print(self.tweets)


    def remove_emojis(self, data):
        emoj = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"  # dingbats
                          u"\u3030"
                          "]+", re.UNICODE)
        return re.sub(emoj, '', data)
    def deEmojify(self, text):
        regrex_pattern = re.compile(pattern = "["
                                              u"\U0001F600-\U0001F64F"  # emoticons
                                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                              "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',text)

    def clean_tweet_text(self, text):
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"_\w+", "", text)
        text = re.sub(r"RT[\s]+", "", text)
        text = re.sub(r"https?:\/\/\S+", "", text)
        # text = text.lower()
        return text

    def cleanTweet(self, tweet):
        # Remove Links, Special Characters etc from tweet
        return " ".join(
            re.sub(
                "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet
            ).split()
        )

    # function to calculate percentage
    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, ".2f")

    # function to generate pie chart. the image will be generate and overwritten every time it runs.

    def plotPieChart(self, positive, negative, neutral):
        fig = plt.figure()
        labels = [
            "Positive [" + str(positive) + "%]",
            "Neutral [" + str(neutral) + "%]",
            "Negative [" + str(negative) + "%]",
        ]
        sizes = [
            positive,
            neutral,
            negative,
        ]
        colors = ["#51d857", "#c2a866", "#d64d4d"]
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.axis("equal")
        plt.tight_layout()
        strFile = "static\images\plot1.png"
        if os.path.isfile(strFile):
            os.remove(strFile)  # Opt.: os.system("rm "+strFile)
        plt.savefig(strFile, transparent=True)
        plt.show()

    # these functions are for generating beautiful and better wordcloud image
    def flatten_list(self, l):
        return [x for y in l for x in y]

    def is_acceptable(self, word: str):
        return word not in stop_words and len(word) > 2

    # Color coding our wordclouds
    def red_color_func(self, word, font_size, position, orientation, random_state=None,**kwargs):
        return f"hsl(0, 100%, {random.randint(25, 75)}%)"

    def green_color_func(self, word, font_size, position, orientation, random_state=None,**kwargs):
        return f"hsl({random.randint(90, 150)}, 100%, 30%)"

    def yellow_color_func(self, word, font_size, position, orientation, random_state=None,**kwargs):
        return f"hsl(42, 100%, {random.randint(25, 50)}%)"

    # Reusable function to generate word clouds
    def generate_word_clouds(self, neg_doc, neu_doc, pos_doc):
        # Display the generated image:
        # collocations = False, is to prevent repeated word to appear in wordcloud
        fig, axes = plt.subplots(1,3, figsize=(20,10))

        wordcloud_neg = WordCloud(max_font_size=50, max_words=100, background_color="white", collocations = False).generate(" ".join(neg_doc))
        axes[0].imshow(wordcloud_neg.recolor(color_func=self.red_color_func, random_state=3), interpolation='bilinear')
        axes[0].set_title("Negative Words", fontsize=40)
        axes[0].axis("off")

        wordcloud_neu = WordCloud(max_font_size=50, max_words=100, background_color="white", collocations = False).generate(" ".join(neu_doc))
        axes[1].imshow(wordcloud_neu.recolor(color_func=self.yellow_color_func, random_state=3), interpolation='bilinear')
        axes[1].set_title("Neutral Words", fontsize=40)
        axes[1].axis("off")

        wordcloud_pos = WordCloud(max_font_size=50, max_words=100, background_color="white", collocations = False).generate(" ".join(pos_doc))
        axes[2].imshow(wordcloud_pos.recolor(color_func=self.green_color_func, random_state=3), interpolation='bilinear')
        axes[2].set_title("Positive Words", fontsize=40)
        axes[2].axis("off")

        plt.tight_layout()
        #     plt.show();
        return fig

    def get_top_percent_words(self, doc, percent):
        # Returns a list of "top-n" most frequent words in a list
        top_n = int(percent * len(set(doc)))
        counter = Counter(doc).most_common(top_n)
        top_n_words = [x[0] for x in counter]
        # print(top_n_words)
        return top_n_words

    def clean_document(self, doc):
        spell = SpellChecker()
        lemmatizer = WordNetLemmatizer()

        # Lemmatize words (needed for calculating frequencies correctly )
        doc = [lemmatizer.lemmatize(x) for x in doc]

        # Get the top 10% of all words. This may include "misspelled" words
        top_n_words = self.get_top_percent_words(doc, 0.1)

        # Get a list of misspelled words
        misspelled = spell.unknown(doc)

        # Accept the correctly spelled words and top_n words
        clean_words = [x for x in doc if x not in misspelled or x in top_n_words]

        # Try to split the misspelled words to generate good words (ex. "lifeisstrange" -> ["life", "is", "strange"])
        words_to_split = [x for x in doc if x in misspelled and x not in top_n_words]
        split_words = self.flatten_list([wordninja.split(x) for x in words_to_split])

        # Some splits may be nonsensical, so reject them ("llouis" -> ['ll', 'ou', "is"])
        clean_words.extend(spell.known(split_words))

        return clean_words

    def get_log_likelihood(self, doc1, doc2):
        doc1_counts = Counter(doc1)
        doc1_freq = {
            x: doc1_counts[x]/len(doc1)
            for x in doc1_counts
        }

        doc2_counts = Counter(doc2)
        doc2_freq = {
            x: doc2_counts[x]/len(doc2)
            for x in doc2_counts
        }

        doc_ratios = {
            # 1 is added to prevent division by 0
            x: math.log((doc1_freq[x] +1 )/(doc2_freq[x]+1))
            for x in doc1_freq if x in doc2_freq
        }

        top_ratios = Counter(doc_ratios).most_common()
        top_percent = int(0.1 * len(top_ratios))
        return top_ratios[:top_percent]

    # Function to generate a document based on likelihood values for words
    def get_scaled_list(self, log_list):
        counts = [int(x[1]*100000) for x in log_list]
        words = [x[0] for x in log_list]
        cloud = []
        for i, word in enumerate(words):
            cloud.extend([word]*counts[i])
        # Shuffle to make it more "real"
        random.shuffle(cloud)
        return cloud
######
    def get_smart_clouds(self, df):

        neg_doc = self.flatten_list(df[df['sentiment']=='negative']['words'])
        neg_doc = [x for x in neg_doc if self.is_acceptable(x)]

        pos_doc = self.flatten_list(df[df['sentiment']=='positive']['words'])
        pos_doc = [x for x in pos_doc if self.is_acceptable(x)]

        neu_doc = self.flatten_list(df[df['sentiment']=='neutral']['words'])
        neu_doc = [x for x in neu_doc if self.is_acceptable(x)]

        # Clean all the documents
        neg_doc_clean = self.clean_document(neg_doc)
        neu_doc_clean = self.clean_document(neu_doc)
        pos_doc_clean = self.clean_document(pos_doc)

        # Combine classes B and C to compare against A (ex. "positive" vs "non-positive")
        top_neg_words = self.get_log_likelihood(neg_doc_clean, self.flatten_list([pos_doc_clean, neu_doc_clean]))
        top_neu_words = self.get_log_likelihood(neu_doc_clean, self.flatten_list([pos_doc_clean, neg_doc_clean]))
        top_pos_words = self.get_log_likelihood(pos_doc_clean, self.flatten_list([neu_doc_clean, neg_doc_clean]))

        # Generate syntetic a corpus using our loglikelihood values
        neg_doc_final = self.get_scaled_list(top_neg_words)
        neu_doc_final = self.get_scaled_list(top_neu_words)
        pos_doc_final = self.get_scaled_list(top_pos_words)

        # Visualise our synthetic corpus
        fig = self.generate_word_clouds(neg_doc_final, neu_doc_final, pos_doc_final)
        return fig


@second.route("/sentiment_logic", methods=["POST", "GET"])
def sentiment_logic():
    # get user input of keyword to search and number of tweets from html form.
    keyword = request.form.get("keyword")
    tweets = request.form.get("tweets")
    sa = SentimentAnalysis()

    # set variables which can be used in the jinja supported html page
    (
        polarity,
        htmlpolarity,
        positive,
        negative,
        neutral,
        keyword1,
        tweet1,
        textArray,
        numspam,
        subjectivity,
        result_color,
        wordcloud_error,
        error_msg
    ) = sa.DownloadData(keyword, tweets)
    print(polarity)
    return render_template(
        "sentiment_analyzer.html",
        polarity=polarity,
        htmlpolarity=htmlpolarity,
        positive=positive,
        negative=negative,
        neutral=neutral,
        keyword=keyword1,
        tweets=tweet1,
        textArray=textArray,
        numspam=numspam,
        subjectivity=subjectivity,
        result_color=result_color,
        wordcloud_error=wordcloud_error,
        error_msg=error_msg
    )

# For zero shot classification model
@second.route("/zeroshot_logic", methods=["POST", "GET"])
def zeroshot_logic():
    # get user input of keyword to search and number of tweets from html form.
    text = request.form.get("text")
    # tweets = request.form.get('tweets')

    sa1 = SentimentAnalysis()
    # textPhrase="hi"
    # set variables which can be used in the jinja supported html page
    label1, label2, label3, score1, score2, score3, text1, resultcolor1, resultcolor2, resultcolor3 = sa1.ZeroShotClassification(
        text
    )
    print(score1)
    return render_template(
        "zeroshot_analyzer.html",
        label1=label1,
        label2=label2,
        label3=label3,
        score1=score1,
        score2=score2,
        score3=score3,
        text=text1,
        resultcolor1=resultcolor1,
        resultcolor2=resultcolor2,
        resultcolor3=resultcolor3
    )


@second.route("/visualize")
def visualize():
    if "user_id" in session:
        return render_template("PieChart.html")
    else:
        return redirect("/")


