
import tensorflow as tf


dexpression = tf.Graph()

with dexpression.as_default():
    with tf.name_scope("Inputs"):
        x = tf.placeholder(tf.uint8, shape=(1, 224, 224, 1), name="x")
        y_ = tf.placeholder(tf.uint8, shape=(1, 7), name="y_")

        # Batch normalization
        #with tf.name_scope("BatchNorm"):
        #    phase = tf.placeholder(dtype=tf.bool, name="phase")
        #    bn = tf.contrib.layers.batch_norm(tf.cast(x, tf.float32), center=True, scale=True, is_training=phase)

        with tf.name_scope("Dropout1"):
            keep_prob1 = tf.placeholder(tf.float32, name="Keep_Prob")
            do1 = tf.nn.dropout(tf.cast(x,tf.float32), keep_prob1, name='Dropout1')

        conv1 = tf.layers.conv2d(inputs=do1, filters=64, kernel_size=(7, 7), strides=(2, 2),
                                 padding='same',  name='conv1')

        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')
        lrn1 = tf.nn.local_response_normalization(input=pool1, name='lrn1')


    with tf.name_scope("Feat_Ex_1"):
        conv2a = tf.layers.conv2d(inputs=lrn1, filters=96, kernel_size=(1, 1), strides=(1, 1),
                                  padding='same', name='conv2a')
        conv2b = tf.layers.conv2d(inputs=conv2a, filters=208, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='conv2b')
        pool2a = tf.layers.max_pooling2d(lrn1, pool_size=(3, 3), strides=(1, 1),  padding='same', name='pool2a')
        conv2c = tf.layers.conv2d(inputs=pool2a, filters=64, kernel_size=(1, 1), strides=(1, 1),
                                  padding='same', name='conv2c')
        concat2 = tf.concat([conv2b, conv2c], axis=3, name='concat2')
        pool2b = tf.layers.max_pooling2d(inputs=concat2, pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2b')

    with tf.name_scope("Feat_Ex_2"):
        conv3a = tf.layers.conv2d(inputs=pool2b, filters=96, kernel_size=(1, 1), strides=(1, 1),
                                  padding='same', name='conv3a')
        pool3a = tf.layers.max_pooling2d(inputs=pool2b, pool_size=(3, 3), strides=(1, 1), padding='same', name='pool3a')
        conv3b = tf.layers.conv2d(inputs=conv3a, filters=208, kernel_size=(3, 3), strides=(1, 1),
                                  padding='same', name='conv3b')
        conv3c = tf.layers.conv2d(inputs=pool3a, filters=64, kernel_size=(1, 1), strides=(1, 1),
                                  padding='same', name='conv3c')
        concat3 = tf.concat([conv3b, conv3c], axis=3, name='concat3')
        pool3b = tf.layers.max_pooling2d(inputs=concat3, pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3b')

        # Dropout Layer
        with tf.name_scope("Dropout2"):
            keep_prob2 = tf.placeholder(tf.float32, name="Keep_Prob")
            do2 = tf.nn.dropout(pool3b, keep_prob2, name='Dropout')

    # Dropout Layer
    # with tf.name_scope("Dropout"):
        # keep_prob = tf.placeholder(tf.float32, name="Keep_Prob")
        # h_drop = tf.nn.dropout(bn, keep_prob, name='Dropout')

    # Batch normalization
     #with tf.name_scope("BatchNorm"):
     #   phase = tf.placeholder(dtype=tf.bool, name="phase")
     #   bn = tf.contrib.layers.batch_norm(do2, center=True, scale=True, is_training=phase)

    with tf.name_scope("Classifier"):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape=shape,  stddev=0.1)
            return tf.Variable(initial)


        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        # Reshape incoming multidimensional tensor to be flat,
        # so we can create a fully connected layer with just 1 dimension
        reshaped = tf.reshape(do2, shape=(1, 272*14*14), name='reshaped')
        y = tf.layers.dense(inputs=reshaped, units=7, name='y')

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    learn_rate = tf.placeholder(dtype=tf.float64, shape=(),name="learning_rate")
    train_step = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cross_entropy)
    prediction = tf.argmax(y, 1)



