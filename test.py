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

    saver = tf.train.import_meta_graph('./GGNN-FL-model-best/model_hd_ant-early-xiaxin-hidden-256-alpha-0.25/ant-9600.meta')
    saver.restore(sess,'./GGNN-FL-model-best/model_hd_ant-early-xiaxin-hidden-256-alpha-0.25/ant-9600')
    gragh = tf.get_default_graph()
    tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]
    feature = gragh.get_tensor_by_name('Placeholder:0')
    adj = gragh.get_tensor_by_name('Placeholder_1:0')
    mask = gragh.get_tensor_by_name('Placeholder_2:0')
    y = gragh.get_tensor_by_name('Placeholder_3:0')
    c = gragh.get_tensor_by_name('Placeholder_4:0')

    outputs = gragh.get_tensor_by_name('readoutlayer_1/add_4:0')
    embedding = gragh.get_tensor_by_name('graphlayer_1/add_20:0')
    feed_dict = {feature:test_adj, adj: test_feature ,mask:test_mask,y:test_y}
    pred = sess.run(outputs,feed_dict)
    pred = np.argmax(pred,1)
    test_y = np.argmax(test_y,1)
    correct_prediction = np.equal(pred, test_y)
    accuracy_all = correct_prediction.astype(np.float32)
    acc = np.mean(accuracy_all)
    auc = metrics.roc_auc_score(test_y, pred)
    print("total:",len(test_y))
    test_f1 = f1_score(test_y, pred, average='binary', pos_label=0)
    print("test_f1="+"{:.3f}".format(test_f1))
    print("acc="+"{:.3f}".format(acc))
    print("auc="+"{:.3f}".format(auc))

