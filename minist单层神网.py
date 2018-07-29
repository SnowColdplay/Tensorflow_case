"使用tf训练神网的基本步骤"
#1.定义网络结构(权重w,偏置b)
#2.定义损失函数，优化器
#3.模型初始化
#4.训练

#导入数据
from tensorflow.examples.tutorials.mnist import input_data
minist = input_data.read_data_sets('path/MINIST_data/', one_hot=True)
print(minist.train.images.shape)

#y=wx+b
#weight和bais都初始化为0
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x, W) + b)
# y = tf.matmul(x, W) + b

#定义损失函数
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy

#定义优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#初始化模型
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 训练
for i in range(1000):
    batch_xs, batch_ys = minist.train.next_batch(100)    #每次随机抽取100个样本，随机梯度下降
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if(i%100==0):
        print(i)

# 验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: minist.test.images,
                                    y_: minist.test.labels}))


