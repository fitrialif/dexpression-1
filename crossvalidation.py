import utils
import glob
import os
import random
import time

DATA_DIR = "../data/resized"


class NFoldCV(object):
    '''
    This creates a class to handle n-fold Cross Validation as described in  
    https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation
    It assumes that data files (images) are in DATA_DIR
    '''
    def __init__(self, folds):
        self.numfolds =folds
        file_path = os.path.join(DATA_DIR, "*")
        # store a list of the files
        self.files = glob.glob(file_path)
        # get the total number of files.
        self.count = len(self.files)

        random.shuffle(self.files)
        # break down the files.
        # self.fold will contain a list of self.numfold lists of files to be used for validation.
        # each list of files is expected to be of length around (total files / numfolds)
        skip = self.numfolds
        self.folds = [self.files[i::skip] for i in xrange(self.numfolds)]

        # self.trainsets will contain a list of self.numfold lists of files to be used for training.
        # each self.trainset[i] is expected to be of length (total files - self.fold[i])
        self.trainsets = [[file for file in self.files if file not in self.folds[j]] for j in range(len(self.folds))]


    def showfolds(self):
        '''
        just a help function to check if the logic in __init__ is sound
        :return: 
        '''
        for i in xrange(self.numfolds):
            fold = self.folds[i]
            trainset= self.trainsets[i]
            print "Fold {}: length is:{}".format(i, len(fold))
            print "Trainset {}: length is:{}".format(i, len(trainset))
            assert(len(fold)+len(trainset) == self.count)


    def getBatch(self,index):
        '''
        This function returns the validation set (data and labels) and training set (data and labels) for a given fold index.
        :param index: fold index.. (index is a number from 1 to self.numfolds)
        :return: 
        '''
        #callfilestoTFData to convert the list of files to a list of arrays(data) and labels(one-hot arrays)
        vdata, vlabels = utils.filestoTFdata(self.folds[index])
        tdata, tlabels = utils.filestoTFdata(self.trainsets[index])
        return vdata, vlabels, tdata, tlabels

    def getStats(self):
        '''
             function to do sanity check
        :return: 
        '''
        print "File Count is {}. Some files are listed below:".format(self.count)
        for i,f in enumerate(self.files[1:10]):
            print "{}: {}".format(i,f)


if __name__ == '__main__':
    ''' 
     test the class. check to see how long it takes to grab a batch
    '''
    cx = NFoldCV(10)
    cx.getStats()
    cx.showfolds()
    print "Start:{}".format(time.strftime("%H:%M:%S"))
    vdata, vlabels, tdata, tlabels = cx.getBatch(1)
    print "vdata_len {} vlabels_len {} tdata_len {} tlabels_len {}".format(len(vdata), len(vlabels), len(tdata), len(tlabels))
    print "End:{}".format(time.strftime("%H:%M:%S"))
    print "done"
