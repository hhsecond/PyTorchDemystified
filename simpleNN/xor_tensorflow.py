import tensorflow as tf

epoch = 501
lr = 1
XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_Y = [[0, 1], [1, 0], [1, 0], [0, 1]]

tf.set_random_seed(10000)
x_ = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 2], name='y-input')
w1 = tf.Variable(tf.random_uniform([2, 5], -1, 1), name="w1")
w2 = tf.Variable(tf.random_uniform([5, 2], -1, 1), name="w2")

b1 = tf.Variable(tf.zeros([5]), name="b1")
b2 = tf.Variable(tf.zeros([2]), name="b2")

a2 = tf.sigmoid(tf.matmul(x_, w1) + b1)
Hypothesis = tf.matmul(a2, w2) + b2
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=Hypothesis, labels=y_))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
prediction = tf.argmax(tf.nn.softmax(Hypothesis), 1)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
        if i % 500 == 0:
            out_cost = sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y})
            print(out_cost)

    for x in XOR_X:
        pred = sess.run(prediction, feed_dict={x_: [x]})
        print(x, pred)
