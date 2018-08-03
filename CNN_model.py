import tensorflow as tf
def read_and_decode(filename):  # 读入tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [100, 100, 3])  # reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    return img, label


#权重和偏置函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积层和池化层
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None,30000])
y_ = tf.placeholder(tf.float32, [None, 2])
x_image = tf.reshape(x, [-1, 100, 100, 3])

"定义卷积层"
"卷积层-非线性激活层-池化层"
w_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"卷积层-非线性激活层-池化层"
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

"全连接层"
w_fc1 = weight_variable([25 * 25 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 25 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

"dropout层"
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

"dropout层后面连接一个softmax层,得到概率输出"
w_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

"定义损失函数和优化器"
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

"准确率函数"
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"训练并验证"
train_path="girl_train.tfrecords"
test_path="girl_test.tfrecords"

img_train, labels_train = read_and_decode(train_path)
img_test, labels_test = read_and_decode(test_path)
# 定义模型训练时的数据批次
img_train_batch, labels_train_batch = tf.train.shuffle_batch([img_train, labels_train],
                                                             batch_size=20,
                                                             capacity=55,
                                                             min_after_dequeue=50)

img_test_batch, labels_test_batch = tf.train.shuffle_batch([img_test, labels_test],
                                                             batch_size=20,
                                                             capacity=55,
                                                             min_after_dequeue=50)

train_labels = tf.one_hot(labels_train_batch, 2, 1, 0)
test_labels = tf.one_hot(labels_test_batch, 2, 1, 0)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(100):
        img_xs, label_xs = sess.run([img_train_batch, train_labels])  # 读取训练 batch
        sess.run(train_step, feed_dict={x_image: img_xs, y_: label_xs, keep_prob: 0.75})
        if i % 10 == 0:
            img_test_xs, label_test_xs = sess.run([img_test_batch, test_labels])  # 读取测试 batch
            acc = sess.run(accuracy, feed_dict={x_image: img_test_xs, y_: label_test_xs, keep_prob: 0.75})
            # acc = sess.run(accuracy, feed_dict={x: img_test_xs, y: label_test_xs, keep_prob: 1.0})
            print("step %d, test accuracy %g" % (i, acc))
    coord.request_stop()
    coord.join(threads)




