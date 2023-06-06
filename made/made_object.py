from made.mask_generator import MaskGenerator
from made.model import ModelMADE
from modules.layers import *
from tensorflow.keras.layers import Input, Layer

class MADE(object):
    def __init__(self,  units_per_layer, natural_input_order, num_masks, order_agn, connectivity_weights, direct_input = False, seed = "42", input_order_seed = None, global_weights = None, only_federated = False, adaptive_factor = 3, kernel_initializer = None, num_tasks = 1):
        self.units_per_layer = units_per_layer
        self.natural_input_order = natural_input_order
        self.num_masks = num_masks
        self.order_agn = order_agn
        self.connectivity_weights = connectivity_weights
        self.direct_input = direct_input
        self.mask_generator = MaskGenerator(num_masks, units_per_layer, natural_input_order, seed, input_order_seed)
        self.global_weights = global_weights
        self.only_federated = only_federated
        self.adaptive_factor = adaptive_factor
        self.kernel_initializer = kernel_initializer
        self.num_tasks = num_tasks

    def build_model(self):
        a = Input(shape = (self.units_per_layer[0],))
        x_layers = []

        #build masks
        masks = self.mask_generator.shuffle_masks()
        direct_mask = None

        for i in range(1,len(self.units_per_layer)-1): #exclude input & output layer
            if i == 1:
                x_layers.append(FedweitDirectInputConnectConditionningMaskedLayer(units = self.units_per_layer[i], mask = masks[i-1], masked_layer_id = i-1, use_cond_mask = self.connectivity_weights, kernel_initializer=self.kernel_initializer, global_weights = self.global_weights, adaptive_factor = self.adaptive_factor, num_tasks = self.num_tasks)(a)) #activation is relu, call custom_layer with previous layer as input-param
            else:
                x_layers.append(FedweitDirectInputConnectConditionningMaskedLayer(units = self.units_per_layer[i], mask = masks[i-1], masked_layer_id = i-1, use_cond_mask = self.connectivity_weights, kernel_initializer=self.kernel_initializer, global_weights = self.global_weights, adaptive_factor = self.adaptive_factor, num_tasks = self.num_tasks)(x_layers[i-2])) #activation is relu, call custom_layer with previous layer as input-param


        #build output layer, output layer's activation is sigmoid.
        if self.direct_input:
            direct_mask = self.mask_generator.get_direct_mask()
            output_layer = FedweitDirectInputConnectConditionningMaskedLayer(units = self.units_per_layer[-1], mask = masks[-1], masked_layer_id = len(self.units_per_layer)-2, activation='sigmoid', use_cond_mask = self.connectivity_weights, direct_mask = direct_mask, kernel_initializer=self.kernel_initializer, global_weights = self.global_weights, adaptive_factor = self.adaptive_factor, num_tasks = self.num_tasks)([x_layers[-1], a])
        else:
            output_layer = FedweitDirectInputConnectConditionningMaskedLayer(units = self.units_per_layer[-1], mask = masks[-1], masked_layer_id = len(self.units_per_layer)-2, activation='sigmoid', use_cond_mask = self.connectivity_weights, direct_mask = direct_mask, kernel_initializer=self.kernel_initializer, global_weights = self.global_weights, adaptive_factor = self.adaptive_factor, num_tasks = self.num_tasks)(x_layers[-1])
        x_layers.append(output_layer)

        #self.model = Model(inputs = a, outputs = x_layers[-1])
        self.model = ModelMADE(inputs = a, outputs = x_layers[-1], mask_generator = self.mask_generator, order_agn = self.order_agn, conn_agn = self.num_masks>1, direct_input = self.direct_input, only_federated = self.only_federated, connectivity_weights = self.connectivity_weights)
        return self.model, masks

    def summary(self):
        return self.model.summary()
