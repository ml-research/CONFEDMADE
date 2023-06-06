import tensorflow as tf
from keras import metrics
import numpy as np
from tensorflow.keras.models import Model


# outputs: output Layer   ---------- Both needed when using ----------
# inputs: input Layer     ----------    base keras.Model    ----------
# mask_generator: Mask Generator instance that manages the Models Masks
# order_agn: Boolean defining if training should be order_agnostic
# conn_agn: Boolean defining if training should be connectivity_agnostic
# direct_input: Boolean defining if direct input masks should be used
class ModelMADE(tf.keras.Model):
    def __init__(self, inputs, outputs, mask_generator, order_agn, conn_agn, direct_input, only_federated, connectivity_weights, **kwargs):
      super(ModelMADE, self).__init__(inputs = inputs, outputs = outputs, **kwargs)
      self.mask_generator = mask_generator
      self.order_agn = order_agn
      self.conn_agn = conn_agn
      self.direct_input = direct_input
      self.only_federated = only_federated
      self.connectivity_weights = connectivity_weights
      #self.resample_every = resample_every
      #self.made = made

    def get_weights(self, type = None):
        weights = {}
        if type == "global":
            weights["W_global"] = []
            if self.connectivity_weights:
                weights["U_global"] = []
            if self.direct_input:
                weights["D_global"] = []
            for counter, layer in enumerate(self.layers[1:]):
                weights["W_global"].append(layer.get_weights("W_global"))
                if self.connectivity_weights:
                    weights["U_global"].append(layer.get_weights("U_global"))
                if self.direct_input and counter == len(self.layers)-2: #starts at layer 1 instead of 0 so it has to be subtracted by 2
                    weights["D_global"].append(layer.get_weights("D_global"))

        elif type == "mask":
            weights["W_mask"] = []
            if self.connectivity_weights:
                weights["U_mask"] = []
            if self.direct_input:
                weights["D_mask"] = []
            for counter, layer in enumerate(self.layers[1:]):
                weights["W_mask"].append(layer.get_weights("W_mask"))
                if self.connectivity_weights:
                    weights["U_mask"].append(layer.get_weights("U_mask"))
                if self.direct_input and counter == len(self.layers)-2:
                    weights["D_mask"].append(layer.get_weights("D_mask"))


        elif type == "prev_masks":
            if len(self.layers[-1].get_weights("W_prev_masks")) == 0:
                return []
            params = ["W"]
            if self.connectivity_weights:
                params.append("U")
            if self.direct_input:
                params.append("D")

            for param in params:
                weights[f"{param}_prev_masks"] = []
                for counter, layer in enumerate(self.layers[1:]):
                    if param == "D":
                        if counter == len(self.layers)-2:
                            weights[f"{param}_prev_masks"].append(layer.get_weights(f"{param}_prev_masks"))
                    else:
                        weights[f"{param}_prev_masks"].append(layer.get_weights(f"{param}_prev_masks"))

        elif type == "all_adapts":
            params = ["W"]
            if self.connectivity_weights:
                params.append("U")
            if self.direct_input:
                params.append("D")

            for param in params:
                weights[f"{param}_all_adapts"] = []
                for counter, layer in enumerate(self.layers[1:]):
                    if param == "D":
                        if counter == len(self.layers)-2:
                            weights[f"{param}_all_adapts"].append(layer.get_weights(f"{param}_all_adapts"))
                    else:
                        weights[f"{param}_all_adapts"].append(layer.get_weights(f"{param}_all_adapts"))

        elif type == "adapts_last_task":
            if len(self.layers[-1].get_weights("W_all_adapts")) == 1:
                return []
            params = ["W"]
            if self.connectivity_weights:
                params.append("U")
            if self.direct_input:
                params.append("D")
            for param in params:
                weights[f"{param}_adapts_last_task"] = []
                for counter, layer in enumerate(self.layers[1:]):
                    if param == "D":
                        if counter == len(self.layers)-2:
                            weights[f"{param}_adapts_last_task"].append(layer.get_weights(f"{param}_adapts_last_task"))
                    else:
                        weights[f"{param}_adapts_last_task"].append(layer.get_weights(f"{param}_adapts_last_task"))


        elif type == "global_last_task":
            if len(self.layers[-1].get_weights("W_all_adapts")) == 1:
                return []
            params = ["W"]
            if self.connectivity_weights:
                params.append("U")
            if self.direct_input:
                params.append("D")
            for param in params:
                weights[f"{param}_global_last_task"] = []
                for counter, layer in enumerate(self.layers[1:]):
                    if param == "D":
                        if counter == len(self.layers)-2:
                            weights[f"{param}_global_last_task"].append(layer.get_weights(f"{param}_global_last_task"))
                    else:
                        weights[f"{param}_global_last_task"].append(layer.get_weights(f"{param}_global_last_task"))

        elif type == "adaptive": #for building kb
            weights["W_adaptive"] = []
            if self.connectivity_weights:
                weights["U_adaptive"] = []
            if self.direct_input:
                weights["D_adaptive"] = []
            for counter, layer in enumerate(self.layers[1:]):
                weights["W_adaptive"].append(layer.get_weights("W_adaptive"))
                if self.connectivity_weights:
                    weights["U_adaptive"].append(layer.get_weights("U_adaptive"))
                if self.direct_input and counter == len(self.layers)-2:
                        weights["D_adaptive"].append(layer.get_weights("D_adaptive"))


        elif type == "atten": #for debugging/monitoring only
            weights["W_atten"] = []
            if self.connectivity_weights:
                weights["U_atten"] = []
            if self.direct_input:
                weights["D_atten"] = []
            for counter, layer in enumerate(self.layers[1:]):
                weights["W_atten"].append(layer.get_weights("W_atten"))
                if self.connectivity_weights:
                    weights["U_atten"].append(layer.get_weights("U_atten"))
                if self.direct_input and counter == len(self.layers)-2:
                    weights["D_atten"].append(layer.get_weights("D_atten"))


        else:
            weights['W'] = []
            weights['bias'] = []
            for l in self.layers[1:]:
                weights['W'].append(l.get_weights(type = "W"))
                weights['bias'].append(l.get_weights(type = "bias"))
            if self.connectivity_weights:
                weights['U'] = []
                for l in self.layers[1:]:
                    weights['U'].append(l.get_weights(type = "U"))
            if self.direct_input:
                weights['D'] =  [self.layers[-1].get_weights(type = "D")]
        return weights

    def set_weights(self, weights, direct_input = "disabled", type = "global"):
        if self.only_federated:
            for count, l in enumerate(self.layers[1:]):
                l.set_weights(weights["W"][count], type = "W")
                l.set_weights(weights["bias"][count], type = "bias")
            if self.connectivity_weights:
                for count, l in enumerate(self.layers[1:]):
                    l.set_weights(weights["U"][count], type = "U")
            if self.direct_input:
                self.layers[-1].set_weights(weights["D"][0], type = "D")
        else:
            if type == "global":
                for count, l in enumerate(self.layers[1:]):
                    temp = {}
                    for key in weights:
                        if key == "D_global":
                            if l == self.layers[-1]: #iterating through layers-1 elements so las indice will bei layers-2
                                temp[key] = weights[key][0]
                        else:
                            temp[key] = weights[key][count]
                    l.set_fedweit_weights(temp, type)
            elif type == "kb":
                for count, l in enumerate(self.layers[1:]):
                    temp = {}
                    for key in weights:
                        if key == "D_adaptive":
                            if count == len(self.layers)-2: #iterating through layers-1 elements so las indice will bei layers-2
                                temp[key] = []
                                [temp[key].append(kb_weight[0]) for kb_weight in weights[key]]
                        else:
                            temp[key] = []
                            [temp[key].append(kb_weight[count]) for kb_weight in weights[key]]
                    l.set_fedweit_weights(temp, type)


    # Method called by fit for every batch
    def train_step(self, data, lossf = None, extended_log = False):
        if self.order_agn:
            # order agnostic and connectivity agnostic training
            if self.conn_agn:
                self.mask_generator.shuffle_inputs(return_mask = False)
                new_masks = self.mask_generator.shuffle_masks()
                for hidden_layer_id in range(len(new_masks)):
                    self.layers[1+hidden_layer_id].set_mask(new_masks[hidden_layer_id]) #assign layer+1 since the first layer is no hidden layer and has no mask

            # order agnostic but not connectivity agnostic training
            else:
                self.layers[1].set_mask(self.mask_generator.shuffle_inputs())
            if self.direct_input:
                self.layers[-1].set_mask(self.mask_generator.get_direct_mask(), direct=True)

        # not order agnostic but connectivity agnostic training
        elif self.conn_agn:
            new_masks = self.mask_generator.shuffle_masks()
            for hidden_layer_id in range(len(new_masks)):
                self.layers[1+hidden_layer_id].set_mask(new_masks[hidden_layer_id])


        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
          y_pred = self(x, training=True)  # Forward pass
          # Compute the loss value
          # (the loss function is configured in `compile()`)
          if lossf == None:
              loss = self.compiled_loss(x=y, x_decoded_mean=y_pred, regularization_losses=self.losses)
          else:
              loss = lossf(y, y_pred)

        #if extended_log:
        #    for var in self.trainable_variables:
        #        print(var.name)
        #        if var.name == 'fedweit_direct_input_connect_conditionning_masked_layer_1/W_adapts:0':
        #            print(var.numpy())
        #Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
