import crossvalidation as cv
import dexpression as dx
import tensorflow as tf
import time
import numpy as np
f10cv = cv.NFoldCV(10)

with tf.Session(graph=dx.dexpression) as sess:

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./temp/PA1/1")
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    ave_acc = 0

    vset, vlabels, tset, tlabels = f10cv.getBatch(1)
    print "@ {}: Run#:{}, Fold#:{}".format(time.strftime("%H:%M:%S"), 0, 1)
    # training
    for i in range(len(tset)):
        # for each data point in the trainingset, feed into network
        dx.train_step.run(feed_dict={dx.x: tset[i], dx.y_: tlabels[i],  })
        s = sess.run(merged_summary, feed_dict={dx.x: tset[i], dx.y_: tlabels[i]})
        # writer.add_summary(s, j*10+i)
    acc = 0
    # test
    conflog = open("conf0.log", "w")
    for i in range(len(vset)):
        # for each data point in the validation set, feed into network, check accuracy
        # accuracy will either be "1.0" or "0.0" depending if the label matches the output
        pred = dx.prediction.eval(feed_dict={dx.x: vset[i], dx.y_: vlabels[i]})

        p = int(pred[0])
        label = int(np.argmax(vlabels[i]))

        if p == label:
            acc +=1
        conflog.write("{}\t{}\t{}\n".format(p, label, "correct" if p == label else "incorrect"))
    acc = acc*1.0/len(vset)
    # get and print the average across the validation set
    print ("acc:{:00.2f}".format(acc*100))
    conflog.write("acc:{:00.2f}\n".format(acc*100))
    conflog.close()

    # accumulate the accuracies to measure the average for this epoch
    ave_acc += acc
    tf.summary.scalar("accuracy", acc)

print "@ {}: Done".format(time.strftime("%H:%M:%S"))
