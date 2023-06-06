from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf
from keras import backend as k
from tensorflow.keras.layers import Input, Layer

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
#from tensorflow.python.keras.activations import tf_activations
import numpy as np

class MaskedLayer(Layer):
    def __init__(self,
                units,
                mask,
                activation='relu',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                global_weights = None,
                **kwargs):
      self.units = units
      self.mask = mask
      self.activation = activations.get(activation)
      self.kernel_initializer = kernel_initializer
      self.bias_initializer = bias_initializer
      self.global_weights = global_weights
      super(MaskedLayer, self).__init__(**kwargs)

    def build(self, input_shape):

      if self.global_weights is not None:
          self.W = tf.Variable(self.global_weights['W'][self.masked_layer_id], trainable=True, name='W')
          self.bias = tf.Variable(self.global_weights['bias'][self.masked_layer_id], trainable=True, name='bias')
      else:
          self.W = self.add_weight(shape=self.mask.shape,
                                      initializer=self.kernel_initializer,
                                      name='W')

          self.bias = self.add_weight(shape=(self.units,),
                                          initializer=self.bias_initializer,
                                          name='bias')

      self.built = True

    def call(self, inputs):
        ## Modified keras.Dense to account for the mask
        masked_weights = self.W*self.mask
        output = k.dot(inputs, masked_weights)
        output = k.bias_add(output, self.bias, data_format = 'channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def set_mask(self, mask):
        self.mask = mask

    def get_mask(self):
        return self.mask

    def compute_output_shape(self, input_shape):
        ##Same as keras.Dense
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)



class ConditionningMaskedLayer(MaskedLayer):
    def __init__(self,
                units,
                mask,
                activation='relu',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                use_cond_mask=False,
                **kwargs):
        self.use_cond_mask = use_cond_mask
        super(ConditionningMaskedLayer, self).__init__(units,
                mask,
                activation,
                kernel_initializer,
                bias_initializer, **kwargs)

    def build(self, input_shape):
        if self.use_cond_mask:
            if self.global_weights is not None:
                self.U = tf.Variable(self.global_weights['U'][self.masked_layer_id], trainable=True, name='U')
            else:
                self.U = self.add_weight(shape=self.mask.shape,
                                         initializer=self.kernel_initializer,
                                         name='U')
        super().build(input_shape)

    def call(self, inputs):
        if self.use_cond_mask == False:
          return super().call(inputs)
        masked_w_weights = self.W*self.mask
        masked_u_weights_times_one_vec = k.dot(tf.ones(tf.shape(inputs)),self.U*self.mask)
        weighted_input = k.dot(inputs, masked_w_weights)
        weighted_input_and_bias = k.bias_add(weighted_input, self.bias, data_format = 'channels_last')
        output = weighted_input_and_bias + masked_u_weights_times_one_vec
        if self.activation is not None:
            output = self.activation(output)
        return output



class DirectInputConnectConditionningMaskedLayer(ConditionningMaskedLayer):
      def __init__(self,
                   units,
                   mask,
                   activation='relu',
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   use_cond_mask=False,
                   direct_mask = None,
                   **kwargs):
        self.direct_mask = direct_mask
        super(DirectInputConnectConditionningMaskedLayer, self).__init__(units,
                mask,
                activation,
                kernel_initializer,
                bias_initializer,
                use_cond_mask,
                **kwargs)

      def build(self, input_shape):
          if self.direct_mask is not None:
              if self.global_weights is not None:
                  self.D = tf.Variable(self.global_weights['D'][0], trainable=True, name='D')
              else:
                  self.D = self.add_weight(shape=self.direct_mask.shape,
                                  initializer=self.kernel_initializer,
                                  name='D')
          super().build(input_shape)

      def set_mask(self, mask, direct = False):
          if direct:
              self.direct_mask = mask
          else:
              super().set_mask(mask)

      def get_mask(self, direct = False):
          if direct:
              return self.direct_mask
          else:
              return super().get_mask

      def get_weights(self, type):
          if type == "W":
              return self.W.value()
          elif type == "bias":
              return self.bias.value()
          elif type == "U":
              return self.U.value()
          elif type == "D":
              return self.D.value()

      def set_weights(self, weights, type):
          if type == "W":
               self.W.assign(weights)
          elif type == "bias":
              self.bias.assign(weights)
          elif type == "U":
              self.U.assign(weights)
          elif type == "D":
              self.D.assign(weights)

      def call(self, inputs):
        if self.direct_mask is None:
            return super().call(inputs)
        input, direct_input = inputs[0], inputs[1]

        masked_w_weights = self.W*self.mask
        weighted_input = k.dot(input, masked_w_weights)
        weighted_input_and_bias = k.bias_add(weighted_input, self.bias, data_format = 'channels_last')
        weighted_direct_input = k.dot(direct_input, self.D * self.direct_mask)
        if self.use_cond_mask:
            masked_u_weights_times_one_vec = k.dot(tf.ones(tf.shape(input)),self.U*self.mask)
            output = weighted_direct_input + weighted_input_and_bias + masked_u_weights_times_one_vec
        else: output = weighted_direct_input + weighted_input_and_bias
        if self.activation is not None:
            output = self.activation(output)
        return output

class FedweitDirectInputConnectConditionningMaskedLayer(DirectInputConnectConditionningMaskedLayer):
      def __init__(self,
                   units,
                   mask,
                   masked_layer_id,
                   activation='relu',
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   use_cond_mask=False,
                   direct_mask = None,
                   adaptive_factor = 3,
                   num_tasks = 1,
                   **kwargs):
        self.masked_layer_id = masked_layer_id
        self.adaptive_factor = adaptive_factor
        self.num_tasks = num_tasks
        self.curr_task = 0
        super(FedweitDirectInputConnectConditionningMaskedLayer, self).__init__(units,
                mask,
                activation,
                kernel_initializer,
                bias_initializer,
                use_cond_mask,
                direct_mask,
                **kwargs)

      def build(self, input_shape):
          if self.global_weights is not None:
              if not "W_global" in self.global_weights:
                  super().build(input_shape)
              else:
                  # init vars
                  self.bias = self.add_weight(shape=(self.units,),
                                                  initializer=self.kernel_initializer,
                                                  name='bias')

                  self.W_global = tf.Variable(self.global_weights['W_global'][self.masked_layer_id], trainable=True, name='W_global')
                  self.W_mask = self.add_weight(shape=(self.units,),
                                                  initializer=self.kernel_initializer,
                                                  name='W_mask')
                  adapts_value = np.zeros(shape=(self.num_tasks, self.W_global.shape[0], self.W_global.shape[1]))
                  adapts_value[0] = np.array(self.W_global.value()/3)
                  self.W_adapts = tf.Variable(adapts_value, dtype="float32", trainable=True, name='W_adapts')

                  # init parameters to store constants
                  self.W_kb = []
                  self.W_adapts_last_task = []
                  self.W_global_last_task = []
                  self.W_prev_masks = []

                  if self.use_cond_mask:
                      # init vars
                      self.U_global = tf.Variable(self.global_weights['U_global'][self.masked_layer_id], trainable=True, name='U_global')
                      self.U_mask = self.add_weight(shape=(self.units,),
                                                      initializer=self.kernel_initializer,
                                                      name='U_mask')
                      adapts_value = np.zeros(shape=(self.num_tasks, self.U_global.shape[0], self.U_global.shape[1]))
                      adapts_value[0] = np.array(self.U_global.value()/3)
                      self.U_adapts = tf.Variable(adapts_value, dtype="float32", trainable=True, name='U_adapts')

                      # init parameters to store constants
                      self.U_kb = []
                      self.U_adapts_last_task = []
                      self.U_global_last_task = []
                      self.U_prev_masks = []
                  if self.direct_mask is not None:
                      # init vars
                      self.D_global = tf.Variable(self.global_weights["D_global"][0], trainable = True, name = "D_global")
                      self.D_mask = self.add_weight(shape=(self.units,),
                                                      initializer=self.kernel_initializer,
                                                      name='D_mask')
                      adapts_value = np.zeros(shape=(self.num_tasks, self.D_global.shape[0], self.D_global.shape[1]))
                      adapts_value[0] = np.array(self.D_global.value()/3)
                      self.D_adapts = tf.Variable(adapts_value, dtype="float32", trainable=True, name='D_adapts')

                      # init parameters to store constants
                      self.D_kb = []
                      self.D_adapts_last_task = []
                      self.D_global_last_task = []
                      self.D_prev_masks = []

                  self.built = True

      def init_task_specific_params(self):
          # should be used when one model is used for all tasks
          # init mask and adaptive params, init atten if kb is given (not at first task)
          # chick if bias is set. if not, the model is still being built and no prev weights should be stored
          self.curr_task += 1

          #self.W_atten = tf.Variable(self.kernel_initializer((len(self.W_kb),)).numpy(), trainable = True, name = f'{self.name}/W_atten')
          self.W_atten = tf.Variable(tf.zeros((len(self.W_kb),)), trainable = True, name = f'{self.name}/W_atten')

          #a new task is to be learned. store previous values relevant to prevent catastrophic forgetting
          self.bias.assign(self.kernel_initializer((self.units, )).numpy())
          self.W_global_last_task = self.W_global.value()
          self.W_adapts_last_task = self.W_adapts.value()[:self.curr_task]
          self.W_prev_masks.append(self.W_mask.value())
          self.W_mask.assign(self.kernel_initializer((self.units, )).numpy())
          temp = self.W_adapts.numpy()
          temp[self.curr_task] = self.W_global.numpy()/3
          self.W_adapts.assign(temp)
          #if self.W_kb != []:
            #  temp = self.W_atten.value()
             # self.W_atten.set_shape((self.num_tasks,))
              #self.W_atten.assign(self.kernel_initializer((self.num_tasks,)).numpy())

          if self.use_cond_mask:
              #self.U_atten = tf.Variable(self.kernel_initializer((len(self.U_kb),)).numpy(), trainable = True, name = f'{self.name}/U_atten')
              self.U_atten = tf.Variable(tf.zeros((len(self.W_kb),)), trainable = True, name = f'{self.name}/U_atten')
              self.U_mask.assign(self.kernel_initializer((self.units, )).numpy())
              temp = self.U_adapts.numpy()
              temp[self.curr_task] = self.U_global.numpy()/3
              self.U_adapts.assign(temp)
              #if self.U_kb != []:
                #  self.U_atten.assign(self.kernel_initializer((self.num_tasks,)).numpy())
              self.U_global_last_task = self.U_global.numpy()
              self.U_adapts_last_task = self.U_adapts.value()[:self.curr_task]
              self.U_prev_masks.append(self.U_mask.value())

          if self.direct_mask is not None:
              #self.D_atten = tf.Variable(self.kernel_initializer((len(self.D_kb),)).numpy(), trainable = True, name = f'{self.name}/D_atten')
              self.D_atten = tf.Variable(tf.zeros((len(self.W_kb),)), trainable = True, name = f'{self.name}/D_atten')
              self.D_mask.assign(self.kernel_initializer((self.units, )).numpy())
              temp = self.D_adapts.numpy()
              temp[self.curr_task] = self.D_global.numpy()/3
              self.D_adapts.assign(temp)
              #if self.D_kb != []:
                #  self.D_atten.assign(self.kernel_initializer((self.num_tasks,)).numpy())
              self.D_global_last_task = self.D_global.numpy()
              self.D_adapts_last_task = self.D_adapts.value()[:self.curr_task]
              self.D_prev_masks.append(self.D_mask.value())



#      def init_atten(self):
#          if self.W_kb != []:
#              self.W_atten = tf.Variable(self.kernel_initializer((len(self.W_kb),)).numpy(),trainable = True, name = "W_atten")
#          if self.use_cond_mask:
#              if self.U_kb != []:
#                  self.U_atten = tf.Variable(self.kernel_initializer((len(self.U_kb),)).numpy(),trainable = True, name = "U_atten")
#          if self.direct_mask:
#              if self.D_kb != []:
#                  self.D_atten = tf.Variable(self.kernel_initializer((len(self.D_kb),)).numpy(),trainable = True, name = "D_atten")


      def get_weights(self, type):                      # FedWeIT Loss function equivalent - comments on elements of W param also apply to U and D
          if type == "W_global":                        # B_t
              return self.W_global.value()
          elif type == "W_mask":                        # m_t
              return self.W_mask.value()
          elif type == "W_global_last_task":            # B_t-1
              return self.W_global_last_task
          elif type == "W_adapts_last_task":            # delta A_i = current_adapts - adapts_snapshotted_at_last_task: clients adaptives for all  previous task, snapshotted at previous timestep
              return self.W_adapts_last_task
          elif type == "W_prev_masks":                  # m_1..t-1 all prev masks for Lamda_2 loss term
              return self.W_prev_masks
          elif type == "W_all_adapts":                  # A_1...t All of the clients task adaptives for Lamda_1 loss term
              return self.W_adapts.value()
          elif type == "W_adaptive":                    # for debugging (already represented in theta thats composed & called for cross entropy loss)
              return self.W_adapts.value()[self.curr_task]
          elif type == "W_atten":                       # for debugging (already represented in theta thats composed & called for cross entropy loss)
              return self.W_atten.value()
          elif type == "W_kb":                          # for debugging (already represented in theta thats composed & called for cross entropy loss)
              return self.W_kb

          elif type == "U_global":
              return self.U_global.value()
          elif type == "U_mask":
              return self.U_mask.value()
          elif type == "U_global_last_task":
              return self.U_global_last_task
          elif type == "U_adapts_last_task":
              return self.U_adapts_last_task
          elif type == "U_prev_masks":
              return self.U_prev_masks
          elif type == "U_all_adapts":
              return self.U_adapts.value()
          elif type == "U_adaptive":
              return self.U_adapts.value()[self.curr_task]
          elif type == "U_atten":
              return self.U_atten.value()
          elif type == "U_kb":
              return self.U_kb

          elif type == "D_global":
              return self.D_global.value()
          elif type == "D_mask":
              return self.D_mask.value()
          elif type == "D_global_last_task":
              return self.D_global_last_task
          elif type == "D_adapts_last_task":
              return self.D_adapts_last_task
          elif type == "D_prev_masks":
              return self.D_prev_masks
          elif type == "D_all_adapts":
              return self.D_adapts.value()
          elif type == "D_adaptive":
              return self.D_adapts.value()[self.curr_task]
          elif type == "D_atten":
              return self.D_atten.value()
          elif type == "D_kb":
              return self.D_kb
          else:
              return super().get_weights(type)


      def set_fedweit_weights(self, weights, _type):
          if _type == "global":
              self.W_global.assign(weights["W_global"])
              if self.use_cond_mask:
                  self.U_global.assign(weights["U_global"])
              if self.direct_mask is not None:
                  self.D_global.assign(weights["D_global"])
          elif _type == "kb": # equals a new task
              for counter, kb_weight in enumerate(weights["W_adaptive"]):
                  self.W_kb.append(tf.Variable(kb_weight, trainable = False, name = f"W_kb_{self.curr_task}_{counter}")) # Note that curr_task is not updated yet

              if self.use_cond_mask:
                  for counter, kb_weight in enumerate(weights["U_adaptive"]):
                      self.U_kb.append(tf.Variable(kb_weight, trainable = False, name = f"U_kb_{self.curr_task}_{counter}"))

              if self.direct_mask is not None:
                  for counter, kb_weight in enumerate(weights["D_adaptive"]):
                      self.D_kb.append(tf.Variable(kb_weight, trainable = False, name = f"D_kb_{self.curr_task}_{counter}"))
              #self.init_atten()
              self.init_task_specific_params()

          else:
              super().set_weights(weights, type)

      def call(self, inputs):
        #reassemble parameters
        #MADE parameters: W and bias (normal), U (connectivity_weights), D (direct input weights)
        if "W_global" in self.global_weights:
            #temp = self.W_global * tf_activations.sigmoid(self.W_mask)
            self.W = self.W_global * tf.keras.activations.sigmoid(self.W_mask) + self.W_adapts[self.curr_task]     #
            if self.W_kb is not None:
                for counter, kb_weight in enumerate(self.W_kb):
                    self.W += self.W_atten[counter] * kb_weight
            if self.use_cond_mask:
                self.U = self.U_global * tf.keras.activations.sigmoid(self.U_mask) + self.U_adapts[self.curr_task]
                if self.U_kb is not None:
                    for counter, kb_weight in enumerate(self.U_kb):
                        self.U += self.U_atten[counter] * kb_weight
            if self.direct_mask is not None:
                self.D = self.D_global * tf.keras.activations.sigmoid(self.D_mask) + self.D_adapts[self.curr_task]
                if self.D_kb is not None:
                    for counter, kb_weight in enumerate(self.D_kb):
                        self.D += self.D_atten[counter] * kb_weight
        return super().call(inputs)

