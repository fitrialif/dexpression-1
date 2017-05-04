import crossvalidation as cv
import dexpression as dx
import tensorflow as tf
import time
import numpy as np
import shutil
import os
f10cv = cv.NFoldCV(10)

accs = []
times = []
learning_rate = 0.000135
do_keep_prob1 = 1.00
do_keep_prob2 = 1.00
num_epochs = 6
# remove folder to clean up for tensorboard
shutil.rmtree("./temp")
for fold in range(10):
    with tf.Session(graph=dx.dexpression) as sess:
        saver = tf.train.Saver()
        tf.summary.scalar("cross_entropy", dx.cross_entropy)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./temp/PA1/{}".format(fold))
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        vset, vlabels, tset, tlabels = f10cv.getBatch(1)
        sz = len(tset)
        # training
        print "{}: Start training on fold#{} a set of {} data points".format(time.strftime("%H:%M:%S"), fold, len(tset))
        start = time.time()
        for e in range(num_epochs):
            for t, i in enumerate(range(len(tset))):
                # for each data point in the trainingset, feed into network
                dx.train_step.run(feed_dict={dx.x: tset[i], dx.y_: tlabels[i], dx.learn_rate: learning_rate,
                                             dx.keep_prob1: do_keep_prob1,
                                             dx.keep_prob2: do_keep_prob2,
                                             })
                s = sess.run(merged_summary, feed_dict={dx.x: tset[i], dx.y_: tlabels[i], dx.learn_rate: learning_rate,
                                                        dx.keep_prob1: do_keep_prob1,
                                                        dx.keep_prob2: do_keep_prob2,
                                                        })
                writer.add_summary(s, global_step=t)

        print "@ {}: Done training".format(time.strftime("%H:%M:%S"))
        duration = time.time() - start
        acc = 0
        # test
        conflog = open("conf{}.log".format(fold), "w")
        print "{}: Start validaion on a set of {} data points".format(time.strftime("%H:%M:%S"), len(vset))
        for i in range(len(vset)):
            # for each data point in the validation set, feed into network, check accuracy
            # accuracy will either be "1.0" or "0.0" depending if the label matches the output
            pred = dx.prediction.eval(feed_dict={dx.x: vset[i], dx.y_: vlabels[i], dx.learn_rate : learning_rate,
                                                 dx.keep_prob1:1.00,
                                                 dx.keep_prob2:1.00,
                                                 })

            p = int(pred[0])
            label = int(np.argmax(vlabels[i]))

            if p == label:
                acc += 1
            conflog.write("{}\t{}\t{}\n".format(p, label, "correct" if p == label else "incorrect"))
        acc = acc*1.0/len(vset)
        # get and print the average across the validation set
        print ("acc:{:00.2f}".format(acc*100))
        conflog.write("acc:{:00.2f}\n".format(acc*100))
        conflog.close()

        # accumulate the accuracies to measure the average for this epoch
        accs.append("{:0.2f}".format(acc*100))
        times.append("{:0.2f}".format(round(duration, 0)))
        folder = "CPfiles/CP{}".format(fold)
        if not os.path.exists(folder):
            os.makedirs(folder)
        checkpoint_file = folder + "/ckpt.sess"
        saver.save(sess, checkpoint_file)
# for fold in range(10):
print "num epochs:{}".format(num_epochs)
print "learning rate is {}".format(learning_rate)
ave_acc = 0.0
for a in accs:
    ave_acc += float(a)

ave_acc /= float(len(accs))
print "@ {}: Done:\n accuracies{}\n times:{}".format(time.strftime("%H:%M:%S"), accs, times)
print "Dropout1 is at {:0.2f}".format(do_keep_prob1)
print "Dropout2 is at {:0.2f}".format(do_keep_prob2)
#print "Has BatchNorm at input"
print " Average accuracy is {:0.2f}".format(ave_acc)
