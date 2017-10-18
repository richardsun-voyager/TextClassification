# TextClassification
Text classification is an important field of Natural Language Processing(NLP), it can be used to automatically label news, text, comments in order to facilitate online information retrieval. This project aims to explore efficient end-to-end models to classify texts in English accurately.

Text classification invovles three steps:
- Preprocessing, that is, how to make the texts clean, for example, we can remove the punctuation and typos, or we can take them as signs. 
- Text represention, that is, how to transform texts into a form that can be handled by classification algorithms directly. Usually, a text consists of words and punctuation which are symbols only readable for humans. Computers only handle scalars, vectors and matrix. 
- Classification, how to classify texts accurately. There are varieties of models, such as decision trees, naive bayesian, SVM and deep neural networks. We need to tune and compare the performances of those models.

In this project, we chose to remove punctuation and typos of texts and split those words. Then, we have tried three methods to extract features, to represent a text by TfIdf values of words, to represent a text by topic models, and to represent a text by artificial neural networks. On the basis of extracted features, we have applied varieties of models, such as decision trees, naive bayesian, SVM and deep neural networks to do classification.

The dataset is 20newsgroup, quite messy originally. Actually there are approximately 500 samples for each category, not sufficient for deep learning. And the final results have shown that traditional bag-of-word model along with multi-layer perceptrons work better than simple deep learning models.

The base model is Stanford 20newsgroup classifier, the link: https://nlp.stanford.edu/wiki/Software/Classifier/20_Newsgroups .

The required packages:
- Tensorflow 1.3
- scikit-learn 0.19
- matplotlib

Platform:
- Anaconda4.3
- Python 3.5.2
- Win10

Hardware:
- Nvidia GTX 960
- i5 CPU
