import numpy as np
import pyts
from pyts.classification import KNeighborsClassifier
import pdb

#print("pyts: {0}".format(pyts.__version__))
def load_data(direc,dataset):
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
    data_test = np.loadtxt(datadir+'_TEST',delimiter=',')
    X_train, X_test, y_train, y_test = [],[], [], []

    y_train = data_train[:,0]-1
    y_test = data_test[:,0]-1

    for index, x in enumerate(data_train):
        X_train.append(data_train[index][1:74])
    X_train = np.array( X_train)

    for index, x in enumerate(data_test):
        X_test.append(data_test[index][1:74])
    X_test = np.array(X_test)
    return X_train, X_test, y_train, y_test

PATH = '/Users/apple/Desktop/dev/projectlife/data/UCR/' # Change this value if necessary

dataset_list = ["Projectlife"]

warping_window_list = [0.03, 0., 0., 0.03, 0.05, 0.06]


error_ed_list = []
error_dtw_list = []
error_dtw_w_list = []
#default_rate_list = []

for i, (dataset, warping_window) in enumerate(zip(dataset_list, warping_window_list)):
    print("Dataset: {}".format(dataset))

    # file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN"
    # file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST"

    # train = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
    # test = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

    # X_train, y_train = train[:, 1:], train[:, 0]
    # X_test, y_test = test[:, 1:], test[:, 0]

    direc = '/Users/apple/Desktop/dev/projectlife/data/UCR'
    summaries_dir = '/Users/apple/Desktop/dev/projectlife/data/logs'

    """Load the data"""
    X_train,X_test,y_train,y_test = load_data(direc,dataset='Projectlife')

    clf_ed = KNeighborsClassifier(metric='euclidean')
    clf_dtw = KNeighborsClassifier(metric='dtw')
    clf_dtw_w = KNeighborsClassifier(metric='dtw_sakoechiba',
                                     metric_params={'window_size': warping_window})

    # Euclidean Distance
    error_ed = 1 - clf_ed.fit(X_train, y_train).score(X_test, y_test)
    print("Error rate with Euclidean Distance: {0:.4f}".format(error_ed))
    error_ed_list.append(error_ed)

    # Dynamic Time Warping
    error_dtw = 1 - clf_dtw.fit(X_train, y_train).score(X_test, y_test)
    print("Error rate with Dynamic Time Warping: {0:.4f}".format(error_dtw))
    error_dtw_list.append(error_dtw)
    print(clf_dtw.predict(X_test))

    # Dynamic Time Warping with a learned warping window
    error_dtw_w = 1- clf_dtw_w.fit(X_train, y_train).score(X_test, y_test)
    print("Error rate with Dynamic Time Warping with a learned warping "
          "window: {0:.4f}".format(error_dtw_w))
    error_dtw_w_list.append(error_dtw_w)
    print(clf_dtw_w.predict(X_test))

    print()
    #
    pdb.set_trace()