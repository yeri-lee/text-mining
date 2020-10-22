import numpy as np
from nltk import RegexpTokenizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import sys


def indexifyData(text, vocabs):
    """
    change to lower case,
    split by words,
    and assign numeric value to each
    """
    text = text.lower()
    tokenizer = RegexpTokenizer(r"\w+")
    words = tokenizer.tokenize(text)
    indexList = []
    for word in words:
        if word in vocabs:
            indexList.append(vocabs[word])
    return indexList


def getVocabs(fname="SMSSpamCollection.txt", custom=False):
    """
    create a dictionary of words,
    isolate spam and ham from each sentence
    assign index for each
    """
    print("Getting Vocabs..")
    out = {}
    with open(fname) as f:
        index = 0
        if custom:
            lines = f.readlines()
            for line in lines:
                line = line.lower().strip()
                if line not in out.keys():
                    line[word] = index
                    index += 1
        else:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i].lower()[1:]
                sentence = " ".join(line.split())
                tokenizer = RegexpTokenizer(r"\w+")
                words = tokenizer.tokenize(sentence)
                for word in words:
                    if word not in out.keys():
                        out[word] = index
                        index += 1
    print("Vocab Count: {}".format(len(out.keys())))
    print("Vocab import complete")
    return out


def features(indexList, vocabs):
    """
    created a table for each vocabs
    """
    lengthOfVocab = len(vocabs)
    x = np.zeros((lengthOfVocab, 1))
    for i in indexList:
        x[i] = 1
    return x


def train_test(fname, custom):
    """
    split train set and dev set
    randomize, and put into numpy to reshape the dimension
    set seed at 1001 and test size of 40%
    """
    print("Preparing Data...")
    X = []
    y = []
    vocabs = getVocabs(fname, custom)
    with open(fname) as f:
        for line in f.readlines():
            label = line.lower().split()[0]
            sent = " ".join(line.lower().split()[1:])
            indexList = indexifyData(sent, vocabs)
            x = features(indexList, vocabs)
            X.append(x)
            y.append(label)
    X = np.array(X).reshape(len(y), len(vocabs)).astype(int)
    y = np.array(y).astype(str)
    print("Prepare complete")
    return train_test_split(X, y, test_size=0.4, shuffle=True, random_state=1001)


def model(fname="SMSSpamCollection.txt", custom=False, NaiveBayes=False):
    """
    created a training model based on Naive Bayes method - multinomial
    tried different types of navice bayes,
    but multinomial had the highest accuracy

    also standardize with sigmoid
    """
    print("start model process.")
    X_train, X_test, Y_train, Y_test = train_test(fname, custom)
    if NaiveBayes:
        clf = MultinomialNB()
    else:
        clf = svm.SVC(kernel="sigmoid", tol=1e-3, max_iter=-1, verbose=1)
    print("\nstarting to train model...\n")
    clf.fit(X_train, Y_train)
    print("\ntrain complete!\n")
    from sklearn.metrics import accuracy_score

    pred = clf.predict(X_test)
    accuracy = accuracy_score(pred, Y_test)
    pred_train = clf.predict(X_train)
    accuracy_train = accuracy_score(pred_train, Y_train)
    """
    finding the accuracy of the test 
    to reduce overfitting and underfitting 
    """
    print("\naccuracy score on train data: ", accuracy_train)
    print("accuracy score on test data: ", accuracy, "\n")
    return clf


if __name__ == "__main__":

    print(
        "Welcome to spam classifier model!\nThis is a simple machine learning example using a sigmoid / Naive Bayes(Multinomial) SVM classifier to classify spam messages."
    )
    custom = (
        True
        if input(
            "First, will you be using a custom vocab list? \n (Vocabs need to be on their own individual lines.) (Y/N): "
        )
        == "Y"
        else False
    )
    fname = None
    if custom:
        fname = input(
            "What is the name of the file? (Must be in the same directory with this script ): "
        )
    naive = (
        True
        if input("Will you be using the NaiveBayes model? (Y/N): ") == "Y"
        else False
    )
    print("\n\nUsing settings:")
    print(
        "Custom: {}\nfile name: {}\nUsing NaiveBayes: {}\n\n".format(
            custom, fname, naive
        )
    )
    if fname == None:
        model = model(NaiveBayes=naive)
    else:
        model = model(fname=fname, custom=True, NaiveBayes=naive)
    print(
        "Now, you can check for your own spam mails!\nBest performance will be when the words are already trained by the model(In the dataset and in the vocab list)."
    )
    print("First loading vocab list...")
    vocab = getVocabs()
    while True:
        text = input("\n\ninput your message here:\n")
        try:
            indexlist = indexifyData(text, vocab)
            fea = features(indexlist, vocab)
            print("This message is a " + model.predict(fea.T)[0] + ".")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("something went wrong:( Try again with a different message.")
