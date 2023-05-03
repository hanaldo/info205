import sys

import pandas as pd
import numpy as np
import time
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def clean_tweet_text(text):
    # TODO Clean the tweet text to remove unnecessary information, such as @s, links and #s
    return text


def analyze_text_to_sentiment(model, tokenizer, text):
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    return ranking[2]  # last result is top result


def try_read(starting_row, total_row):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # time taken to read data
    s_time = time.time()
    df = pd.read_csv("tweets.csv", engine="python", nrows=total_row, skiprows=range(1, starting_row))
    e_time = time.time()
    print("Reading time: ", (e_time - s_time), "seconds")

    s_time = time.time()

    labels = ["-1", "0", "1"]
    new_col_values = []
    for index, row in df.iterrows():
        print(index)  # Show me the progress!
        senti = analyze_text_to_sentiment(model, tokenizer, clean_tweet_text(row["text"]))
        new_col_values.append(labels[senti])

    # Add the new column to the dataframe
    df["sentiment"] = new_col_values
    df.to_csv("updated_data_with_sentiment.csv", index=False)

    e_time = time.time()
    print("Finished analyzing time: ", (e_time - s_time), "seconds")


if __name__ == "__main__":
    try_read(int(sys.argv[1]), int(sys.argv[2]))
