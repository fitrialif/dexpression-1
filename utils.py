import os
import numpy as np
import cv2
def filestoTFdata(files):
    '''
    
    :param files:is a list of filenames of format E$_xxxxxxx.png representing images of emotions.
    The $ is a number from 1..7 representing the emotion (anger, contempt, disgust, fear, happy, sadness, surprise)
    
    :return: a list of (image) data, and a list of one-hot-arrays representing emotion labels 
    '''
    images= []
    labels = []
    for f in files:
        # get the filename. extract the E$ to acquire emotion label
        fn = os.path.basename(f)
        emo = int(fn.split('_')[0][1]) - 1
        # create a one-hot vector
        onehot = np.array([int(i==emo) for i in range(7)], dtype=np.uint8)
        # reshape to be as expected i.e. (1,7)
        emo_label = np.reshape(onehot, (1, 7))
        # add to list
        labels.append(emo_label)
        # read the imagee as grayscale
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        # expand to that expected
        expanded_img = np.expand_dims(np.expand_dims(img,0),3)
        # convert to numpy array
        image = np.array(expanded_img)
        # add to list
        images.append(image)
    # return lists of images and labels
    return images, labels

# test the function above
if __name__== "__main__":
    filelist = ['../data/resized/E3--S060_005_00000011.png']
    images, labels = filestoTFdata(filelist)
