import numpy as np
import tensorflow as tf

class MaskGenerator(object):
  # num_masks: The amount of masks that will be cycled through during training. if num_masks == 1 then connectivity agnostic training is disabled
  # units_per_layer = Array containing # of units per layer
  # seed = The seed used for randomly sampling the masks, for guaranteeing reproducability
  # natural_input_order = Boolean defining if the natural input order (x1, x2, x3 etc) should be used
  # current_mask: Integer to keep track of the mask currently used (xth mask)
  # m: The mask values assigned to the networks units. 0 is the index of the input layer, 1 is the index of the first hidden layer and so on
  def __init__(self, num_masks, units_per_layer, natural_input_order = False, seed=42, input_order_seed = None):
    self.num_masks = num_masks
    self.units_per_layer = units_per_layer
    self.seed = seed
    self.natural_input_order = natural_input_order
    self.input_order_seed = input_order_seed
    self.current_mask = 0
    self.m = {}

    if natural_input_order: # init input ordering according to settings
      self.m[0] = np.arange(self.units_per_layer[0])
    else:
      self.shuffle_inputs(return_mask = False)

  #Iterate through the hidden layers, resample new connectivity values m and build/return the resulting new masks
  def shuffle_masks(self):
    layer_amount = len(self.units_per_layer)
    rng = np.random.RandomState(self.seed+self.current_mask)
    self.current_mask = (self.current_mask + 1) % self.num_masks # Cycle through masks
    for i in range(1, layer_amount -1): #skip input layer & output layer and only iterate through hidden_layers
      self.m[i] = rng.randint(self.m[i-1].min(), self.units_per_layer[0] -1, size = self.units_per_layer[i]) # sample m from [min_m(previous_layer, d-1)] for all hidden units
    new_masks = [tf.convert_to_tensor((self.m[l-1][:, None] <= self.m[l][None,:]), dtype=np.float32) for l in range(1, layer_amount-1)] # build hidden layer masks
    new_masks.append(tf.convert_to_tensor((self.m[layer_amount-2][:, None] < self.m[0][None, :]), dtype = np.float32)) #build output layer mask. Note that the m values for the output layer are the same as for the input layer
    return new_masks

  # builds & returns direct mask. Call this method after shuffling inputs if order_agnostic training is active.
  # Note that the Mask values m are the same for both input and output layers
  def get_direct_mask(self):
    return tf.convert_to_tensor((self.m[0][:, None] < self.m[0][None, :]), dtype = np.float32)

  # shuffle input ordering and return new mask for first hidden layer
  def shuffle_inputs(self, return_mask = True):
    if self.input_order_seed is None:
        self.m[0] = np.random.permutation(self.units_per_layer[0])
    else:
        rng = np.random.RandomState(self.input_order_seed)
        self.m[0] = rng.permutation(self.units_per_layer[0])
        self.input_order_seed += 1
    if return_mask:
      return tf.convert_to_tensor((self.m[0][:, None] <= self.m[1][None,:]), dtype=np.float32)
    return
