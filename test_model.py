import dexpression as dx
import crossvalidation as cv
import tensorflow as tf
import numpy as np

model_no = 1
folder = "CPfiles/CP{}".format(model_no)
cp_file = folder + "/ckpt.sess"
f10cv = cv.NFoldCV(10)
accs = []
for fold in range(10):
    with tf.Session(graph = dx.dexpression) as session:
        saver = tf.train.Saver()
        saver.restore(session, cp_file)
        vset, vlabels, tset, tlabels = f10cv.getBatch(fold)

        acc=0
        for i in range(len(vset)):
            # for each data point in the validation set, feed into network, check accuracy
            # accuracy will either be "1.0" or "0.0" depending if the label matches the output
            pred = dx.prediction.eval(feed_dict={dx.x: vset[i], dx.y_: vlabels[i],
                                                 dx.keep_prob1: 1.00,
                                                 dx.keep_prob2: 1.00})

            p = int(pred[0])
            label = int(np.argmax(vlabels[i]))

            if p == label:
                acc +=1

        acc = acc*1.0/len(vset)
        # get and print the average across the validation set
        print ("acc:{:00.2f}".format(acc*100))
        accs.append("{:00.2f}".format(acc*100))

print "validated on model#{} {} sets:{}".format(model_no, 10, accs)
ave_accs = 0.0
for acc in accs:
    ave_accs += float(acc)
print "ave accuracy:{:0.2f}".format(ave_accs/len(accs))



