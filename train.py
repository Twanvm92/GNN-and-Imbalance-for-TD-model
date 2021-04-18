from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score

import pickle as pkl

from utils import *
from run_models import GNN, MLP
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set random seed
# seed = 123
# np.random.seed(seed)
# tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('label',0, 'test_label')  # 0 1
flags.DEFINE_string('save_file', 'hd-argouml-GGNN-FL-early-xiaxin-hidden-256.txt', 'file String.')
flags.DEFINE_string('save_model', "./model_hd_-argouml-GGNN-FL-early-xiaxin-hidden-256/argouml", 'file String.')
flags.DEFINE_string('dataset', 'hd_argouml', 'Dataset string.')
flags.DEFINE_string('model', 'gnn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs',200, 'Number of epochs to train.')
flags.DEFINE_float('alpha', 0.25, 'Initial learning rate.')
flags.DEFINE_integer('batch_size',128 , 'Size of batches per epoch.')
flags.DEFINE_integer('input_dim', 300, 'Dimension of input.')
flags.DEFINE_integer('hidden', 256, 'Number of units in hidden layer.')  # 32, 64, 96, 128
flags.DEFINE_integer('steps', 2, 'Number of graph layers.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('evaluateEvery',100 , 'How many steps are run for validation each time.')

# Load data
train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y,d_lossweight = load_data(
    FLAGS.dataset)
f = open(FLAGS.save_file,'w+',encoding='utf-8')
# Some preprocessing
print('loading training set')
train_adj, train_mask = preprocess_adj(train_adj)
train_feature = preprocess_features(train_feature)
print('loading validation set')
val_adj, val_mask = preprocess_adj(val_adj)
val_feature = preprocess_features(val_feature)
print('loading test set')
test_adj, test_mask = preprocess_adj(test_adj)
test_feature = preprocess_features(test_feature)

if FLAGS.model == 'gnn':
    # support = [preprocess_adj(adj)]
    # num_supports = 1
    model_func = GNN
elif FLAGS.model == 'gcn_cheby':  # not used
    # support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GNN
elif FLAGS.model == 'dense':  # not used
    # support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': tf.placeholder(tf.float32, shape=(None, None, None)),
    'features': tf.placeholder(tf.float32, shape=(None, None, FLAGS.input_dim)),
    'mask': tf.placeholder(tf.float32, shape=(None, None, 1)),
    'labels': tf.placeholder(tf.float32, shape=(None, train_y.shape[1])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'steps_per_epoch': tf.placeholder_with_default(0.0, shape=()),
    'd_lossweight': d_lossweight
}

# Create model
model = model_func(placeholders, input_dim=FLAGS.input_dim, logging=True)

# Initialize session

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
session_conf = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=session_conf)

# sess = tf.Session()
saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)


# Define model evaluation function
def evaluate(features, support, mask, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, mask, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.embeddings, model.preds, model.labels],
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4]


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
min_loss = 100000000
max_f1 = 0
End_pred = None
End_label = None
best_f1 = 0
best_epoch = 0
test_doc_embeddings = None

print('train start...')

# Train model
currentStep = 0
steps_per_epoch = (int)(len(train_y)/FLAGS.batch_size)+1
for epoch in range(FLAGS.epochs):
    t = time.time()

    # Training step
    indices = np.arange(0, len(train_y))
    np.random.shuffle(indices)

    train_loss, train_acc = 0, 0
    for start in range(0, len(train_y), FLAGS.batch_size):
        currentStep += 1
        end = start + FLAGS.batch_size
        idx = indices[start:end]
        # Construct feed dictionary
        feed_dict = construct_feed_dict(train_feature[idx], train_adj[idx], train_mask[idx], train_y[idx], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['steps_per_epoch']: steps_per_epoch})

        outs = sess.run([model.opt_op, model.loss, model.accuracy,model.learning_rate], feed_dict=feed_dict)
        train_loss = outs[1]
        train_acc = outs[2]

        print("Epoch:", '%04d' % (epoch + 1),"Step:{}".format(currentStep),"train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_acc))

        if currentStep % FLAGS.evaluateEvery == 0:
            # Validation
            val_cost, val_acc, val_duration, _, val_pred, val_labels = evaluate(val_feature, val_adj, val_mask, val_y,
                                                                                placeholders)
            val_f1 = f1_score(val_pred, val_labels, average='binary', pos_label=FLAGS.label)

            # Test
            test_cost, test_acc, test_duration, embeddings, pred, labels = evaluate(test_feature, test_adj, test_mask,
                                                                                    test_y,
                                                                                    placeholders)

            test_pre = precision_score(labels, pred, average='binary', pos_label=FLAGS.label)
            test_recall = recall_score(labels, pred, average='binary', pos_label=FLAGS.label)
            test_f1 = f1_score(labels, pred, average='binary', pos_label=FLAGS.label)
            auc = roc_auc_score(labels, pred)

            if test_f1 > best_f1:
                best_f1 = test_f1
                best_epoch = epoch
                test_doc_embeddings = embeddings

            # if val_cost<min_loss:
            if val_f1>max_f1:
                min_loss = val_cost
                max_f1 = val_f1
                End_pred = pred
                End_label = labels
                path = saver.save(sess,FLAGS.save_model , global_step=currentStep)
                print("Saved model checkpoint to {}\n".format(path))
                f.write(str(currentStep)+':'+"Saved model checkpoint")

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
                  "val_loss=", "{:.5f}".format(val_cost),"val_f1=", "{:.3f}".format(val_f1), "test_loss=", "{:.3f}".format(test_cost),"test_pre=", "{:.3f}".format(test_pre), "test_recall=",
                  "{:.3f}".format(test_recall), "test_f1=", "{:.3f}".format(test_f1),"test_acc=", "{:.3f}".format(test_acc),"test_auc=", "{:.3f}".format(auc),
                  "time=", "{:.5f}".format(time.time() - t), "best_f1=", "{:.3f}".format(best_f1))

            result = "Epoch:" + "%04d" % (epoch + 1) + " train_loss=" + "{:.5f} ".format(
                train_loss)  + " val_loss="+ "{:.5f}".format(val_cost)+" val_f1="+ "{:.3f}".format(val_f1) + "test_loss=" + "{:.3f} ".format(
                test_cost) + " test_pre=" + "{:.3f} ".format(
                test_pre) + " test_recall=" + "{:.3f} ".format(
                test_recall) + " test_f1=" + "{:.3f} ".format(
                test_f1) + " test_acc=" + "{:.3f} ".format(
                test_acc) + " test_auc=" + "{:.3f} ".format(
                auc) +" time=" + "{:.5f} ".format(time.time() - t) \
                     + " best_f1=" + "{:.3f} ".format(best_f1)
            f.write(result)
            f.write('\n')

print("Optimization Finished!")

# Best results
print('Best epoch:', best_epoch)
print("Test set results:",
      "f1=", "{:.3f}".format(best_f1))

print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(End_label, End_pred, digits=3))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(End_label, End_pred, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(End_label, End_pred, average='micro'))


# For visualization
doc_vectors = []
for i in range(len(test_doc_embeddings)):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append(str(np.argmax(test_y[i])) + ' ' + doc_vector_str)

doc_embeddings_str = '\n'.join(doc_vectors)
with open(FLAGS.dataset + '_doc_vectors.txt', 'w'):
    f.write(doc_embeddings_str)

