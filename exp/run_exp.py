from time import time
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_imdb(path, shuffle=True, random_state=42, \
              vectorizer = TfidfVectorizer(min_df=5, max_df=1.0,  sublinear_tf=True, use_idf=True)):
    
    print "Loading the imdb reviews data"
    
    train_neg_files = glob.glob(path+"/train/neg/*.txt")
    train_pos_files = glob.glob(path+"/train/pos/*.txt")
    
    train_corpus = []
    y_train = []
    
    for tnf in train_neg_files:
        f = open(tnf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(0)
        f.close()
    
    for tpf in train_pos_files:
        f = open(tpf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(1)
        f.close()
    
    test_neg_files = glob.glob(path+"/test/neg/*.txt")
    test_pos_files = glob.glob(path+"/test/pos/*.txt")
    
    test_corpus = []
    
    y_test = []
    
    for tnf in test_neg_files:
        f = open(tnf, 'r')
        test_corpus.append(f.read())
        y_test.append(0)
        f.close()
    
    for tpf in test_pos_files:
        f = open(tpf, 'r')
        test_corpus.append(f.read())
        y_test.append(1)
        f.close()
    
    print "Data loaded."
    
    print "Extracting features from the training dataset using a sparse vectorizer"
    print "Feature extraction technique is %s." % vectorizer
    t0 = time()
    
    X_train = vectorizer.fit_transform(train_corpus)
    
    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_train.shape
    print
        
    print "Extracting features from the test dataset using the same vectorizer"
    t0 = time()
        
    X_test = vectorizer.transform(test_corpus)
    
    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_test.shape
    print
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_train))       
        
        X_train = X_train.tocsr()
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_corpus_shuffled = [train_corpus[i] for i in indices]
        
        indices = np.random.permutation(len(y_test))
        
        X_test = X_test.tocsr()
        X_test = X_test[indices]
        y_test = y_test[indices]
        test_corpus_shuffled = [test_corpus[i] for i in indices]
    else:
        train_corpus_shuffled = train_corpus
        test_corpus_shuffled = test_corpus
         
    return X_train, y_train, X_test, y_test, train_corpus_shuffled, test_corpus_shuffled

from al.instance_strategies import LogGainStrategy, RandomStrategy, UncStrategy, BootstrapFromEach, QBCStrategy, ErrorReductionStrategy

def _run_a_single_trial(X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget, t, all_performances):

        pool = set(range(len(y_pool)))

        trainIndices = []

        bootstrapped = False

        # Choosing strategy
        if al_strategy == 'erreduct':
            active_s = ErrorReductionStrategy(classifier=classifier_name, seed=t, classifier_args=classifier_arguments)
        elif al_strategy == 'loggain':
            active_s = LogGainStrategy(classifier=classifier_name, seed=t, classifier_args=classifier_arguments)
        elif al_strategy == 'qbc':
            active_s = QBCStrategy(classifier=classifier_name, classifier_args=classifier_arguments)
        elif al_strategy == 'rand':
            active_s = RandomStrategy(seed=t)
        elif al_strategy == 'unc':
            active_s = UncStrategy(seed=t)

        model = None
        
        labels = np.unique(y_pool)

        #Loop for prediction
        while len(trainIndices) < budget and len(pool) >= step_size:

            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
            else:
                newIndices = active_s.chooseNext(pool, X_pool, model, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

            pool.difference_update(newIndices)

            trainIndices.extend(newIndices)

            model = classifier_name(**classifier_arguments)

            model.fit(X_pool[trainIndices], y_pool[trainIndices])

            # Prediction
            y_probas = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            # Measures
            
            all_performances["accuracy"][len(trainIndices)].append(metrics.accuracy_score(y_test, y_pred))
            all_performances["auc"][len(trainIndices)].append(metrics.roc_auc_score(y_test, y_probas[:,1]))
            
            for label in labels:            
                all_performances["precision_"+str(label)][len(trainIndices)].append(metrics.precision_score(y_test, y_pred, pos_label=label))
                all_performances["recall_"+str(label)][len(trainIndices)].append(metrics.recall_score(y_test, y_pred, pos_label=label))
                all_performances["f1_"+str(label)][len(trainIndices)].append(metrics.f1_score(y_test, y_pred, pos_label=label))

from collections import defaultdict

def run_trials(X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget, num_trials):

        all_performances = {}
        
        average_performances = {}
        
        labels = np.unique(y_pool)
        
        measures = ["accuracy", "auc"]
        
        for measure in ["precision_", "recall_", "f1_"]:
            for label in labels:
                measures.append(measure+str(label))
        
        
        for measure in measures:
            all_performances[measure] = defaultdict(list)
            average_performances[measure] = {}

        for t in range(num_trials):
            print "trial", t
            _run_a_single_trial(X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, classifier_arguments, bootstrap_size,  step_size, budget, t, all_performances)
        
        return all_performances



from drms.dr import TruncatedSVDDR

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import argparse

if __name__ == '__main__':
                
    parser = argparse.ArgumentParser()
    #parser.add_argument("-c","--classifier", choices=['KNeighborsClassifier', 'LogisticRegression', 'SVC', 'BernoulliNB',
    #                'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GaussianNB', 'MultinomialNB'],
    #                default='LogisticRegression', help="Represents the classifier that will be used (default: MultinomialNB) .")

    # Classifier's arguments
    #parser.add_argument("-a","--arguments", default='',
    #                help="Represents the arguments that will be passed to the classifier (default: '').")

    # Data: Testing and training already split
    #parser.add_argument("-d", '--data', nargs=2, metavar=('pool', 'test'),
    #                default=["data/imdb-binary-pool-mindf5-ng11", "data/imdb-binary-test-mindf5-ng11"],
    #                help='Files that contain the data, pool and test, and number of features (default: data/imdb-binary-pool-mindf5-ng11 data/imdb-binary-test-mindf5-ng11 27272).')

    # Data: Single File
    #parser.add_argument("-sd", '--sdata', type=str, default='',
    #                help='Single file that contains the data. Cross validation will be performed (default: None).')
    
    # Whether to make the data dense
    #parser.add_argument('-make_dense', default=False, action='store_true', help='Whether to make the sparse data dense. Some classifiers require this.')
    
    # Number of Folds
    #parser.add_argument("-cv", type=int, default=10, help="Number of folds for cross validation. Works only if a single dataset is loaded (default: 10).")

    # File: Name of file that will be written the results
    parser.add_argument("-f", '--file', type=str, default="with-pca-l1",
                    help='This feature represents the name that will be written with the result. If it is left blank, the file will not be written (default: None ).')

    # Number of Trials
    parser.add_argument("-nt", "--num_trials", type=int, default=10, help="Number of trials (default: 10).")

    # Strategies
    parser.add_argument("-st", "--strategies", choices=['erreduct', 'loggain', 'qbc', 'rand','unc'], nargs='*',default=['rand'],
                    help="Represent a list of strategies for choosing next samples (default: rand).")

    # Boot Strap
    parser.add_argument("-bs", '--bootstrap', default=10, type=int,
                    help='Sets the Boot strap (default: 10).')

    # Budget
    parser.add_argument("-b", '--budget', default=500, type=int,
                    help='Sets the budget (default: 500).')

    # Step size
    parser.add_argument("-sz", '--stepsize', default=10, type=int,
                    help='Sets the step size (default: 10).')

    # Sub pool size
    parser.add_argument("-sp", '--subpool', default=None, type=int,
                    help='Sets the sub pool size (default: None).')
    
    X_tr, y_tr, X_te, y_te, tr_corp, te_corp = load_imdb("C:\\Users\\Mustafa\\Desktop\\aclImdb")
    
    from drms.dr import TruncatedSVDDR
    
    tsvdr = TruncatedSVDDR(500)
    tsvdr.fit(X_tr)
    
    X_tr = tsvdr.transform(X_tr)
    X_te = tsvdr.transform(X_te)
    
    args = parser.parse_args()
    
    #clf_args = {}

    #for argument in args.arguments.split(','):
    #    if argument.find('=') >= 0:
    #        variable, value = argument.split('=')
    #        clf_args[variable] = eval(value)
    
    clf = LogisticRegression
    clf_args = {}
    clf_args['penalty'] = 'l1'
    
    average_performances = {}
    
    from front_end.cl.run_al_cl import save_all_results, plot_and_save_average_results
    
    for strategy in args.strategies:
        performances = run_trials(X_tr, y_tr, X_te, y_te, strategy, clf, clf_args, args.bootstrap,  args.stepsize, args.budget, args.num_trials)
        measures = performances.keys()
            
        bs = sorted(performances[measures[0]].keys())
        
        average_performances[strategy] = {}
        
        for measure in measures:
            
            average_performances[strategy][measure] = {}
            for b in bs:
                average_performances[strategy][measure][b] = np.mean(performances[measure][b])         
        
            if args.file is not None:
                file_name = args.file + "_" + strategy + "_" + measure +"_all.csv"
                save_all_results(file_name, performances[measure])
                
                
        
    plot_and_save_average_results(average_performances, args.file)
        