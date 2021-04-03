from __future__ import division
from __future__ import print_function
import csv
import time
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import f1_score
import pickle as pkl
from utils import *
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'hd_apache-ant-1.7.0', 'Dataset string.')  # 'mr','ohsumed','R8','R52'
train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y,d_lossweight = load_data(
    FLAGS.dataset)
test_adj, test_mask = preprocess_adj(test_adj)
test_feature = preprocess_features(test_feature)


with tf.Session() as sess:
    # tf.train.import_meta_graph： 用来加载meta文件中的图,以及图上定义的结点参数包括权重偏置项等需要训练的参数,也包括训练过程生成的中间参数,所有参数都是通过graph调用接口
    saver = tf.train.import_meta_graph('./model/ant-FL/ant-9600.meta')
    # 从文件恢复模型参数tf.train.latest_checkpoint
    saver.restore(sess,'./model/ant-FL/ant-9600')
    # saver.restore(sess,tf.train.latest_checkpoint('./GGNN-FL-model/model_hd_jedit-early-xiaxin-hidden-256-alpha-0.25/'))
    gragh = tf.get_default_graph() #获取当前图，为了后续训练时恢复变了
    # 得到当前图中所有变量的名称
    tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]
    print(tensor_name_list)
    # get_tensor_by_name(name="训练时的参数名称")来获取
    feature = gragh.get_tensor_by_name('Placeholder:0')  # 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
    print(feature.shape)
    adj = gragh.get_tensor_by_name('Placeholder_1:0')  # 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
    print(adj.shape)
    mask = gragh.get_tensor_by_name('Placeholder_2:0')  # 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
    print(mask.shape)
    y = gragh.get_tensor_by_name('Placeholder_3:0')  # 获取输出变量
    print(y.shape)
    c = gragh.get_tensor_by_name('Placeholder_4:0')
    print(c.shape)

    print(test_feature.shape)
    print(test_adj.shape)
    print(test_mask.shape)

    # readoutlayer_1/add_4：0 就是输出的向量outputs
    outputs = gragh.get_tensor_by_name('readoutlayer_1/add_4:0')  # 获取网络输出值
    embedding = gragh.get_tensor_by_name('graphlayer_1/add_20:0') # 获取embeddeding
    print(tf.shape(outputs))
    feed_dict = {feature:test_adj, adj: test_feature ,mask:test_mask,y:test_y}
    # pred = sess.run()
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    pred = sess.run(outputs,feed_dict)
    # embedd= sess.run(embedding,feed_dict)
    # print(np.shape(embedd))
    # test_1 = embedd[633]
    # embedd_1 = []
    # for x in test_1:
    #     a = np.mean(x)
    #     embedd_1.append(a)
    # print(embedd_1)
    # print(len(embedd_1))
    pred = np.argmax(pred,1)
    test_y = np.argmax(test_y,1)
    # for a,i in enumerate(test_y):
    #     if i == 0:
    #         if pred[a] == 0:
    #             print(a)
    correct_prediction = np.equal(pred, test_y)
    accuracy_all = correct_prediction.astype(np.float32)
    acc = np.mean(accuracy_all)
    auc = metrics.roc_auc_score(test_y, pred)
    print("total:",len(test_y))
    test_f1 = f1_score(test_y, pred, average='binary', pos_label=0)
    print("test_f1="+"{:.3f}".format(test_f1))
    print("acc="+"{:.3f}".format(acc))
    print("auc="+"{:.3f}".format(auc))

