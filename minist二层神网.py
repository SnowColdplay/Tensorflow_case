#https://blog.csdn.net/xazzh/article/details/79403670

#导入数据
from tensorflow.examples.tutorials.mnist import input_data
minist = input_data.read_data_sets('path/MINIST_data/', one_hot=True)
print(minist.train.images.shape)


#网络结构
input_node = 784
output_node = 10
layer1_node= 300  #隐藏层节点
batch_size = 100


import tensorflow as tf
x = tf.placeholder(tf.float32, [None, input_node])
keep_prob=tf.placeholder(tf.float32)  #dropout的placeholder
#隐藏层参数
w1 = tf.Variable(tf.truncated_normal([input_node, layer1_node],stddev=0.1))  #权重初始化为正态分布，加噪声，避免relu激活时的0梯度
b1 = tf.Variable(tf.zeros([layer1_node]))
#输出层softmax参数
w2 = tf.Variable(tf.truncated_normal([layer1_node, output_node]))
b2 = tf.Variable(tf.zeros([output_node]))

"定义算法公式"
lay1 = tf.nn.relu(tf.matmul(x, w1) + b1)  #隐藏层
lay1_drop=tf.nn.dropout(lay1,keep_prob)   #dropout层
y=tf.nn.softmax(tf.matmul(lay1_drop,w2)+b2) #输出softmax层

"以上是前向传播，以下定义损失函数和优化器，反向传播过程"
#定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, output_node])

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

#初始化模型
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#训练
for i in range(1000):
    batch_xs, batch_ys = minist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob:0.5})
    if(i%100==0):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        validate_acc = sess.run(accuracy, feed_dict={x: minist.test.images,
                                    y_: minist.test.labels,keep_prob:0.7})
        print("%d %g"%(i,validate_acc))