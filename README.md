# Lazy-Sparse-SGD


Implemented a Python version of L2-regularized logistic regression learning with scalable on-line stochastic gradient descent. Efficient sparse updates were achieved by lazy update of regularization. The hashing trick is used for memory saving. 

The data are articles from DBPedia, and the label is the type of the article. There are in total 17 classes in the dataset, and they are from the first level class in DBpedia ontology. Each document may belong to multiple classes, and we train a separate binary classifier for each class. The data contains one document per line of the format: 

> docID    label1,label2,...    word1 word2 word3...

Given the path to testing dataset, `LR20.py` streams through training data from stdin, and produces output in the following format, one line per test sample:

> label1  probability_label1,label2 probability_label2,...
