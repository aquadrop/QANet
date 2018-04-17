"""
-------------------------------------------------
   File Name：     model
   Description :
   Author :       deep
   date：          18-4-17
-------------------------------------------------
   Change Activity:
                   18-4-17:
                   
   __author__ = 'deep'

   https://openreview.net/pdf?id=B14TlG-RW
-------------------------------------------------
"""
import tensorflow as tf

class QANetModel(object):
    def __init__(self, config):
        self.embedding_size = 300
        self.kernel_size = 7
        self.num_filters = 128
        self.channel = 1
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)
        self.num_layers_in_block = 4
        self.num_attention_heads = 8

    def _create_placeholder(self):
        # batch size, question_len, embedding_size
        self.question_placeholder = tf.placeholder(
            tf.float32, shape=(None, None, self.embedding_size), name='questions')
        # batch size, question_len, embedding_size
        self.context_placeholder = tf.placeholder(
            tf.float32, shape=(None, None, self.embedding_size), name='context/paragraph')
        # batch_size, start, end index
        self.answer_span_placeholder = tf.placeholder(
            tf.float32, shape=(None, 2), name='answer_span')

    def _create_filters(self):
        self.depthwise_filter = tf.get_variable(name="depthwise_filter",
                                           shape=(self.kernel_size, self.kernel_size, self.channel, 1),
                                           dtype=tf.float32,
                                           regularizer=self.regularizer,
                                           initializer=self.initializer)
        self.pointwise_filter = tf.get_variable(name="pointwise_filter",
                                           shape=(1, 1, self.channel, self.num_filters),
                                           dtype=tf.float32,
                                           regularizer=self.regularizer,
                                           initializer=self.initializer)

    def _layer_norm(self, inputs):
        return tf.contrib.layers.layer_norm(inputs)

    def _position_encode(self):
        pass

    def _create_encoder_layer(self, data_inputs):
        inputs = self._layer_norm(data_inputs)
        for _ in range(self.num_layers_in_block):
            outputs = tf.nn.separable_conv2d(inputs,
                                             depthwise_filter=self.depthwise_filter,
                                             pointwise_filter=self.pointwise_filter,
                                             strides=(1,1,1,1),
                                             padding='SAME')
            inputs = outputs
        outputs = inputs
        return outputs

    def _create_encoder_block(self, data_inputs):
        #conv
        outputs = self._create_encoder_layer(data_inputs)
        #self_attention
        inputs = self._layer_norm(outputs)

        #feedforward network


    def build(self):
        pass
