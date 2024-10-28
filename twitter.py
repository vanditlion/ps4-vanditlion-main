"""
Author: Vandit shah
Date: 10/25/2024
Description: PS4 assignment
"""

import numpy as np

from string import punctuation

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.

    Parameters
    --------------------
        fname  -- string, filename

    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.

    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """

    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    np.savetxt(outfile, vec)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.

    Parameters
    --------------------
        infile    -- string, filename

    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """

    word_list = {}
    index = 0
    with open(infile, 'r') as fid:
        ### ========== TODO: START ========== ###
        # part 1-1: process each line to populate word_list
        for line in fid:
            words = extract_words(line)
            for word in words:
                if word not in word_list.keys():
                    word_list[word] = index
                    index = index + 1
        ### ========== TODO: END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.

    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)

    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """

    num_lines = sum(1 for line in open(infile,'r'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))

    with open(infile, 'r') as fid:
        ### ========== TODO: START ========== ###
        # part 1-2: process each line to populate feature_matrix
        for index, line in enumerate(fid):
            extracted_words = extract_words(line)
            for word in extracted_words:
                if word not in word_list:
                    continue
                dict_index = word_list[word]
                
                feature_matrix[index, dict_index] = 1
        ### ========== TODO: END ========== ###
    #print("Feature Matrix - ",feature_matrix)
    return feature_matrix


def test_extract_dictionary(dictionary):
    err = 'extract_dictionary implementation incorrect'

    assert len(dictionary) == 1811

    exp = [('2012', 0),
           ('carol', 10),
           ('ve', 20),
           ('scary', 30),
           ('vacation', 40),
           ('just', 50),
           ('excited', 60),
           ('no', 70),
           ('cinema', 80),
           ('frm', 90)]
    act = [sorted(dictionary.items(), key=lambda it: it[1])[i] for i in range(0, 100, 10)]
    assert exp == act


def test_extract_feature_vectors(X):
    err = 'extract_features_vectors implementation incorrect'

    assert X.shape == (630, 1811)

    exp = np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                    [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    act = X[:10,:10]
    assert (exp == act).all()


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric='accuracy'):
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'

    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 # map points of hyperplane to +1
  
    ### ========== TODO: START ========== ###
    # part 2-1: compute classifier performance with sklearn metrics
    # hint: sensitivity == recall
    # hint: use confusion matrix for specificity (use the labels param)
    
    if metric == 'accuracy':
        score = metrics.accuracy_score(y_true, y_label)
    elif metric == 'f1_score':
        score = metrics.f1_score(y_true, y_label)
    elif metric == 'auroc':
        score = metrics.roc_auc_score(y_true, y_label)
    elif metric == 'precision':
        score = metrics.precision_score(y_true, y_label)
    elif metric == 'sensitivity':
        conf_matrix = metrics.confusion_matrix(y_true, y_label)
        score = conf_matrix[1, 1] / float((conf_matrix[1, 1] + conf_matrix[1, 0]))
    elif metric == 'specificity':
        conf_matrix = metrics.confusion_matrix(y_true, y_label)
        score = conf_matrix[0, 0] / float((conf_matrix[0, 0] + conf_matrix[0, 1]))
    else:
        print("There is an error in the metrics.")
    ### ========== TODO: END ========== ###

    return score

def test_performance():
    """Ensures performance scores are within epsilon of correct scores."""

    y_true = [ 1,  1, -1,  1, -1, -1, -1,  1,  1,  1]
    y_pred = [ 3.21288618, -1.72798696,  3.36205116, -5.40113156,  6.15356672,
               2.73636929, -6.55612296, -4.79228264,  8.30639981, -0.74368981]
    metrics = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']
    scores  = [     3/10.,      4/11.,   5/12.,        2/5.,          2/6.,          1/4.]

    import sys
    eps = sys.float_info.epsilon

    for i, metric in enumerate(metrics):
        assert abs(performance(y_true, y_pred, metric) - scores[i]) < eps, \
            (metric, performance(y_true, y_pred, metric), scores[i])


def cv_performance(clf, X, y, kf, metric='accuracy'):
    """
    Splits the data, X and y, into k folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    scores = []
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make "continuous-valued" predictions
        y_pred = clf.decision_function(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score):
            scores.append(score)
    return np.array(scores).mean()


def select_param_linear(X, y, kf, metric='accuracy'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that maximizes the average k-fold CV performance.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """

    print('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)

    ### ========== TODO: START ========== ###
    # part 2-3: select optimal hyperparameter using cross-validation
    # hint: create a new sklearn linear SVC for each value of C
    score = []
    for C in C_range:
        clf = SVC(C=C, kernel="linear")
        scores = cv_performance(clf, X, y, kf, metric=metric)
        score.append(scores)
    
    print(str(metric) + ':', score)
    max_index = score.index(max(score))
    return C_range[max_index]
    ### ========== TODO: END ========== ###


def select_param_rbf(X, y, kf, metric='accuracy'):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.

    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metric  -- string, option used to select performance measure

    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """

    print('RBF SVM Hyperparameter Selection based on ' + str(metric) + ':')

    ### ========== TODO: START ========== ###
    # (Optional) part 3-1: create grid, then select optimal hyperparameters using cross-validation
    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-3, 3)
    max_score = 0
    best_c = 0
    best_gamma = 0
    for c in C_range:
        for gamma in gamma_range:
            score = cv_performance(SVC(C=c,kernel='rbf',gamma=gamma),X=X,y=y,kf=kf,metric=metric)
            if score > max_score:
                max_score = score
                best_c = c
                best_gamma = gamma
    print(" Max score for ", metric, " = ", max_score)
    print("-------------------------------------------")
    return best_gamma, best_c
    ### ========== TODO: END ========== ###


def performance_CI(clf, X, y, metric='accuracy'):
    """
    Estimates the performance of the classifier using the 95% CI.

    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure

    Returns
    --------------------
        score        -- float, classifier performance
        lower, upper -- tuple of floats, confidence interval
    """

    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric)
    
   
    
  
    ### ========== TODO: START ========== ###
    # part 4-2: use bootstrapping to compute 95% confidence interval
    # hint: use np.random.randint(...) to get a random sample from y
    # hint: lower and upper are the values at 2.5% and 97.5% of the scores
    n = X.shape[0]
    
    B = 1000
    lower = 25
    upper = 975
    scores = []
    
    for i in range(B):
        sampled_data = []
        for j in range(n):
            sampled_data.append(np.random.randint(n))
        
        sampled_pred = clf.decision_function(X[sampled_data])
        
        y_sampled = y[sampled_data]
        score = performance(y_sampled, sampled_pred, metric=metric)
        scores.append(score)
        
    scores.sort()
    
    mean = np.float64(np.mean(scores))
    lower = np.float64(scores[lower])
    upper = np.float64(scores[upper])
    
    
    ### ========== TODO: END ========== ###

    return mean, lower, upper

######################################################################
# main
######################################################################

def main():
    # read the tweets and its labels
    dictionary = extract_dictionary('../data/tweets.txt')
    test_extract_dictionary(dictionary)
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    test_extract_feature_vectors(X)
    y = read_vector_file('../data/labels.txt')

    # shuffle data (since file has tweets ordered by movie)
    X, y = shuffle(X, y, random_state=0)

    # set random seed
    np.random.seed(1234)

    # split the data into training (training + cross-validation) and testing set
    X_train, X_test = X[:560], X[560:]
    y_train, y_test = y[:560], y[560:]

    metric_list = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']

    ### ========== TODO: START ========== ###
    # test_performance()

    # part 2-2: create stratified folds (5-fold CV)
    
    print("-----Linear SVM Hyperparameter Selection-----")
    C_range = 10.0 ** np.arange(-3, 3)
    print("-----C Values-----",C_range)
    sf = StratifiedKFold(shuffle=True,n_splits=5)
    # part 2-4: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    max_score_para = {}
    for metric in metric_list:
        c_max = select_param_linear(X_train, y_train, sf, metric=metric)
        max_score_para[metric] = c_max
    # (Optional) part 3-2: for each metric, select optimal hyperparameter for RBF-SVM using CV
    best_gamma = {}
    for metric in metric_list:
        gamma, best_c = select_param_rbf(X_train, y_train, sf, metric=metric)
        best_gamma[metric] = gamma, best_c
    
    print(best_gamma)
    print("-----------------------------------")
    print("Best C and gamma : ", gamma, best_c)
    print("-----------------------------------")
    
    
    # part 4-1: train linear-kernal SVM with selected hyperparameters
    linear_SVM = SVC(C=1.0, kernel='linear')
    rbf_SVM = SVC(C=100, gamma=0.01, kernel='rbf')
    
    linear_SVM.fit(X_train, y_train)
    rbf_SVM.fit(X_train, y_train)
    

    # part 4-3: use bootstrapping to report performance on test data
    ProcessLookupError
    print("----------- For Linear SVM -----------")
    for metric in metric_list:
        mean, lower, upper = performance_CI(linear_SVM, X_test, y=y_test, metric=metric)
        print(metric, " : ", mean, lower, upper)
    print("----------- For RBF Kernel SVM ----------")   
    for metric in metric_list: 
        mean, lower, upper = performance_CI(rbf_SVM, X_test, y=y_test, metric=metric)
        print(metric, " : ", mean, lower, upper)
    print("------------------------------------")
    # part 5: identify important features (hint: use best_clf.coef_[0])

    coeff = linear_SVM.coef_.ravel()
    print("Coefficients : ", linear_SVM.coef_.ravel())
    
    top_positive_coeff = np.argsort(coeff)[-10:]
    print("Top positive Coefficients : ", top_positive_coeff)
    
    top_negative_coeff = np.argsort(coeff)[:10]
    print("Top negative Coefficients : ",top_negative_coeff)
    
    coefficients = np.hstack(
        [top_negative_coeff, top_positive_coeff]
    )
    print("Top Coefficients\n\n", coefficients)
    
    positive_words = []
    for positive_coeff in top_positive_coeff:
        positive_words.append(list(dictionary.keys())[positive_coeff])
        print(list(dictionary.keys())[positive_coeff])
        
    negative_words = []
    for negative_coeff in top_negative_coeff:
        negative_words.append(list(dictionary.keys())[negative_coeff])
        print(list(dictionary.keys())[negative_coeff])
        
    colors = ['green' if c < 0 else 'blue' for c in coeff[coefficients]]
    plt.bar(np.arange(20), coeff[coefficients], color=colors)
    
    plt.xticks(np.arange(1, 21),
                negative_words + positive_words, rotation=60, ha='right')
    plt.show(block=True)
    ### ========== TODO: END ========== ###

    ### ========== TODO: START ========== ###
    # part 6: (optional) contest!
    # uncomment out the following, and be sure to change the filename
    end_words = ["a", "an", "and", "are", "as", "at", "be", "but", "by","for", "if", "in", "into", "is", "it","no", "not", "of", "on", "or", "such",
                    "that", "the", "their", "then", "there", "these","they", "this", "to", "was", "will", "with"]
    
    def NewDictionary(infile):
        words_list = {}
        index = 0
        with open(infile, 'r') as f:
            for lines in f:
                extract = extract_words(lines)
                for words in extract: 
                    if words not in words_list.keys():
                        if words not in end_words:
                            words_list[words]=index
                            index += 1
            return words_list
    
    def NewVectorFeatures(infile, words_list):
        
        num_lines = sum(1 for line in open(infile, 'r'))
        
        num_words = len(words_list)
        
        feature_matrix = np.zeros((num_lines, num_words))
        
        with open(infile, 'r') as f:
            for index, lines in enumerate(f):
                extract = extract_words(lines)
                for i in extract:
                    if i not in words_list:
                        continue
                    x_index = words_list[i]
                    feature_matrix[index, x_index] = 1
            return feature_matrix
        
    svm_clf = SVC(C=100,kernel='rbf',gamma=0.01)
        
    NewDict = NewDictionary('../data/tweets.txt')
    X = NewVectorFeatures('../data/tweets.txt', NewDict)
    svm_clf.fit(X, y)
        
    X_held = NewVectorFeatures('../data/held_out_tweets.txt', NewDict)
    y_pred = svm_clf.decision_function(X_held)
    y_label = np.sign(y_pred)
    y_label[y_label == 0] = 1
    
    write_label_answer(y_label.astype(np.int_), '../data/vanditushah_twitter.txt')
    
    ### ========== TODO: END ========== ###


if __name__ == '__main__':
    main()
