
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np
def main(argv):
    # 获取数据实例
    mnist = input_data.read_data_sets('/home/python/Desktop/project/ai/nn', one_hot=True)
    # 准备数据占位符，x ->[None, 784]  y_true-->[None, 10]
    with tf.variable_scope("data"):
        # 手写数字特征值
        x = tf.placeholder(tf.float32, [None, 784])#表示输入是一个[?,784]的矩阵
        # 手写数字的目标值
        y_true = tf.placeholder(tf.int32, [None, 10])
    # 准备参数，建立模型，w->[784, 10] b->[10]
    with tf.variable_scope("model"):
        # 权重
        weights = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name="weight")
        # 偏置
        bias = tf.Variable(tf.constant(0.0, shape=[10]), name="bias")
        # 通过矩阵运算得出预测结果[None, 784] *[784, 10] + [10]
        y_predict = tf.matmul(x, weights) + bias
    # softmax回归，以及交叉熵损失计算
    with tf.variable_scope("loss"):
        # labels-->真实值， logits-->预测值
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
    # 梯度下降
    with tf.variable_scope("SGD"):
        train_op = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
    # 计算准确率
    equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    # 收集损失和准确率，权重
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("weght", weights)
    # 变量初始化op
    init_op = tf.global_variables_initializer()
    # 合并变量
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    im = cv2.imread('./images/888.jpg',cv2.IMREAD_GRAYSCALE).astype(np.float32)  
    im = cv2.resize(im,(28,28),interpolation=cv2.INTER_CUBIC)  
        #图片预处理  
        #img_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY).astype(np.float32)  
        #数据从0~255转为-0.5~0.5  
    img_gray =(im-(255/2.0))/255
        #cv2.imshow('out',img_gray)  
        #cv2.waitKey(0)  
    x_img = np.reshape(img_gray , [-1 , 784])
    print(x_img)
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)
        # 建立事件文件
        filewriter = tf.summary.FileWriter('/home/python/Desktop/project/ai/nn', graph=sess.graph)
        # 指定迭代次数去训练
        for i in range(10000):
            mnist_x, mnist_y = mnist.train.next_batch(500)
            # 运行梯度下降op
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
            print("准确率：", sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}))
            # 运行收集op
            summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
            # 写入事件文件
            filewriter.add_summary(summary, i)
        print("测试集里面的准确率：", sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))
        
  
        #print x_img  
        output = sess.run(y_predict , feed_dict = {x:x_img})  
        print(output)
        print(np.argmax(output))
        saver.save(sess, "./model_data/model.ckpt")  
    return None

#
#f weight_variable(shape):
#  w = tf.V#iable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
#  return w

#
#f bias_variable(shape):
#  b = tf.V#iable(tf.constant(0.0, shape=shape))
#  return b

#
#f model():
#  """
#  得出卷积网络模型
#  :return: 数据占位符，预测值
#  """
#  # 准备输入数据占位符, x->[None, 784], y_true->[None, 10]
#  with tf.variable_scope("data"):
#      # 手写数字特征值#
#        x = tf.placeholder(tf.float32, [None, 784])
#
#      # 手写数字的目标值#
#       y_true = tf.placeholder(tf.int32, [None, 10])
#
#  # 卷积层1
#  with tf.variable_scope("conv1"):
#      # 初始化卷积层1 的参数,5*5filter的大小，1是图#入通道，32输出filter数量
#        w_conv1 = weight_variable([5, 5, 1, 32])
#
#      # 初始化偏置#
#        b_conv1 = bias_variable([32])
#
#      # 对输入图片改变形状（卷积要求）#
#        x_shape = tf.reshape(x, [-1, 28, 28, 1])
#
#      # 进行卷积、激活、池化操作
#        x_relu1 = tf.nn.relu(tf.nn.conv2d#_shape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)
#
#        x_pool1 = tf.nn.max_pool#_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
#
#  # 卷积层2
#  with tf.variable_scope("conv2"):
        # 初始化卷积层2 #数,5*5filter的大小，32是上一次卷积之后的输#道，64输出filter数量
#        w_conv2 = weight_variable([5, 5, 32, 64])
#
#      # 初始化偏置#
#        b_conv2 = bias_variable([64])
#
#      # 进行卷积、激活、池化操作
#        x_relu2 = tf.nn.relu(tf.nn.conv2d#_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)
#
#        x_pool2 = tf.nn.max_pool#_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
#
#  # 全连接层1
#  with tf.variable_scope("FC1"):
        # 首先对上一层的结果形状进行改变（全连接层要求）[None, 7, 7, 64] -->[None, #7*64]#
#        x_fc1 = tf.reshape(x_pool2, [-1, 7 * 7 * 64])
#
#      # 初始化参数w,b#
#        w_fc1 = weight_variable([7 * 7 * 64, 1024])
#
#        b_fc1 = bias_variable([1024])
#
#      # 特征加权计算#
#        x_fc1_relu = tf.nn.relu(tf.matmul(x_fc1, w_fc1) + b_fc1)
#
#  # 全连接层2
#  with tf.variable_scope("FC2"):
#      # 初始化权重和偏置，[1024, 10], [10]#
#        w_fc2 = weight_variable([1024, 10])
#
#        b_fc2 = bias_variable([10])
#
#      # 特征加权计算, [None, 1024] * [1024, 10]#
#        y_predict = tf.matmul(x_fc1_relu, w_fc2) + b_fc2
#
#  return x, y_true, y_predict

#
#def main(argv):
#
#    mnist = input_dat#read_data_sets('/home/python/Desktop/project/ai/nn', one_hot=True)
#
#  # 构建好模型，得出输出结果#
#    x, y_true, y_predict = model()
#
#  # 计softmax，交叉熵损失#
#    with tf.variable_scope("compute"):
#
#        loss = tf.reduce_mean(tf.nn.so#max_cross_entropy_with_logits(labels=y_true, logits=y_predict))
#
#  # 梯度下降#
#    with tf.variable_scope("SGD"):
#
#        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#
#  # 计算准确率
#  with tf.variable_scope("acc"):#
#        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
#
#        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
#
#  # 初始化变量#
#    init_op = tf.global_variables_initializer()
#
#  # 会话#
#    with tf.Session() as sess:
#
#      # 初始化#
#        sess.run(init_op)
#
#      # 指定迭代次数
#      for i in range(1000):
#          # 获取数据#
#            mnist_x, mnist_y = mnist.train.next_batch(50)
#
#          # 运行优化器#
#            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
#
#          # 计算准确率
#            print("准确率：# sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}))
#
#    return None


if __name__ == "__main__":
       tf.app.run()
