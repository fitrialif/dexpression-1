
def initconmatrix(numlabels=7):
    ''' 
    Create an empty dictionary representing a confuion matrix.
    The keys are the labels. Each label will also contain a dictionary whose keys are predictions,
    and whose values are counts. So, for example.
    confmatrix['6']['3']= 5, means that there wer 5 instances where the label '6' was predicted as '3'.
    '''
    confmatrix = {}
    for i in range(numlabels):
        confmatrix[i] = {}
        for j in range(numlabels):
            confmatrix[i][j] = 0

def procline(line, confmatrix):
    '''
    This function returns the label, prediction pair read from a confx file.
    :param line: a line from a confx file, typically formatted as "label prediction correct?" or "acc xxxx".
    :return: label, prediction
    '''
    data = line.split(' ')

def checkconfmatrix(cmatrix, acc):
    '''
    Check the confusion matrix by comparing it with a given accuracy.
    :param cmatrix: the confusion matrix
    :param acc: the given accuracy
    :return: 
    '''
    labelcount = len(cmatrix.keys())
    corrects = 0
    total = 0
    for k in cmatrix.keys():
        corrects += cmatrix[k][k]
        for j in cmatrix[k].keys():
            total += cmatrix[k][j]
    comp_acc = float(corrects)/total
    if comp_acc==acc:
        return True
    else:
        return False

def read_confFile(fn):
    cmatrix = initconfmatrix()
    with open(fn, 'r') as cfile:
        for line in cfile.readlines():
            line = split(' ')
            if line[0] == 'acc':
                acc = float(line[1].strip())
                ok = checkconfmatrix(cmatrix, acc)
                assert(ok)
            else:
                procline(line, cmatrix)









def main():
    pass


if __name__== '__main__':
    main()