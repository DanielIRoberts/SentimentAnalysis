# Daniel Roberts dir170130

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

class SentimentAnalysis:
    def __init__(self, dataFile):
        # Preprocessing data
        self.data = pd.read_csv(dataFile)
        self.data = self.data[["airline_sentiment", "airline", "text"]]
        self.data["text"] = self.data.text.map(lambda x: x.lower())

        # Tranforming text
        countVec = CountVectorizer()
        counts = countVec.fit_transform(self.data["text"])
        transformer = TfidfTransformer().fit(counts)
        counts = transformer.transform(counts)

        # Encoding sentiment
        label = LabelEncoder()
        label.fit(self.data["airline_sentiment"])
        self.data["airline_sentiment"] = label.transform(self.data["airline_sentiment"])

        # Splitting data
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(counts, self.data["airline_sentiment"], test_size = .1)

    def model(self, alpha):
        # Fitting model and predicting
        model = MultinomialNB(alpha = alpha).fit(self.xTrain, self.yTrain)
        predicted = model.predict(self.xTest)
        return(np.mean(predicted == self.yTest))

    def question(self):
        # Getting sums
        names = self.data.airline.unique()
        avg = np.zeros(len(names))
        maxVal = 0
        maxName = ""
        for i in np.arange(len(names)):
            index = self.data.index[self.data["airline"] == names[i]]
            avg[i] = self.data.iloc[index, 0].mean()

            if (avg[i] > maxVal):
                maxVal = avg[i]
                maxName = names[i]
        
        print(pd.DataFrame(data = avg, index = names, columns = ["Average"]))
        print("\n" + maxName + ": " + str(maxVal))

if __name__ == "__main__":
    sentAn = SentimentAnalysis(sys.argv[1])
    alpha = []
    acc = []
    count = 0

    # Runs until user chooses to quit
    while True:
        user = input("Enter value for alpha or n to quit:\n")
        if user == "n":
            break

        alpha.append(float(user))
        acc.append(sentAn.model(alpha[count]))

        count += 1

    # Plotting data from model predictions
    plt.plot(alpha, acc, "o")
    plt.ylabel("Accuracy")
    plt.xlabel("Alpha")
    plt.show()

    print(pd.DataFrame(acc, index = alpha, columns = ["Accuracy"]))
    print()

    # Outputting answer to question
    sentAn.question()
        