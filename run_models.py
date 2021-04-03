from layers import *
from run_metrics import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.embeddings = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations = [self.inputs]
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.embeddings = self.activations[-2]
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        # globalStep = tf.Variable(0, name="globalStep", trainable=False)
        print("self.globalStep",self.globalStep)

        # 计算梯度,得到梯度和变量
        gradsAndVars = self.optimizer.compute_gradients(self.loss)
        # 将梯度应用到变量下，生成训练器
        self.opt_op = self.optimizer.apply_gradients(gradsAndVars, global_step=self.globalStep)
        # self.opt_op = self.optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # self.opt_op = self.optimizer.minimize(self.loss,global_step=globalStep)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GNN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GNN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.mask = placeholders['mask']
        self.placeholders = placeholders
        # self.epoch = placeholders['epoch']  # 获取当前epoch 2020 7 20
        self.steps_per_epoch = placeholders['steps_per_epoch']  # 一个epoch多少batch_size 2020 9 20
        self.d_lossweight = placeholders['d_lossweight']  # 获取加权系数 2020 7 22

        self.globalStep = tf.Variable(0, name="globalStep", trainable=False)
        self.learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, self.globalStep,
                                                       self.steps_per_epoch*FLAGS.epochs, 1e-5,
                                                       power=1)

        # self.learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, self.epoch,
        #                                           FLAGS.epochs, 1e-4,
        #                                           power=1)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        print('build...')
        self.build()

    def _loss(self):
        # Weight decay loss
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for var in tf.trainable_variables():
            if 'weights' in var.name or 'bias' in var.name:
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error

        # self.loss += focal_loss_weight(self.outputs, self.placeholders['labels'],self.d_lossweight)
        # self.loss += weighted_cross_entropy(self.outputs, self.placeholders['labels'],self.d_lossweight)
        # self.loss += focal_loss_fixed_new(self.outputs, self.placeholders['labels'])
        # self.loss += my_focal_loss_fixed_new(self.outputs, self.placeholders['labels'])
        self.loss += one_focal_loss_fixed_new(self.outputs, self.placeholders['labels'],self.d_lossweight) # 正在使用
        # self.loss += DSC_loss(self.outputs, self.placeholders['labels'])
        # self.loss += ghm_class_loss(self.outputs, self.placeholders['labels'])
        # self.loss += dice_loss(self.outputs, self.placeholders['labels'])
        # self.loss += xiaxin_weighted_cross_entropy(self.outputs, self.placeholders['labels'],self.d_lossweight)
        # self.loss += tversky_loss(self.outputs, self.placeholders['labels'])
        # self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])
        self.preds = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):

        self.layers.append(GraphLayer(input_dim=self.input_dim,
                                      output_dim=FLAGS.hidden,
                                      placeholders=self.placeholders,
                                      act=tf.tanh,
                                      sparse_inputs=False,
                                      dropout=True,
                                      steps=FLAGS.steps,
                                      logging=self.logging))

        self.layers.append(ReadoutLayer(input_dim=FLAGS.hidden,
                                        output_dim=self.output_dim,
                                        placeholders=self.placeholders,
                                        act=tf.tanh,
                                        sparse_inputs=False,
                                        dropout=True,
                                        logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)