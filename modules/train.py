import pdb
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as tf_metrics

from misc.utils import *

class TrainModule:
    """ Class module used for train CONFEDMADE clients
        Also serves the purpose of initializing, loading and saving states for individual client in FL.
    """
    def __init__(self, args, logger, nets):
        self.args = args
        self.logger = logger
        self.nets = nets
        self.init_atten = None
        self.metrics = {
            'train_lss': tf_metrics.Mean(name='train_lss'),
            'train_acc': tf_metrics.CategoricalAccuracy(name='train_acc'),
            'valid_lss': tf_metrics.Mean(name='valid_lss'),
            'valid_acc': tf_metrics.CategoricalAccuracy(name='valid_acc'),
            'test_lss' : tf_metrics.Mean(name='test_lss'),
            'test_acc' : tf_metrics.CategoricalAccuracy(name='test_acc')
        }

    def init_state(self, cid):
        self.state = {
            'client_id': cid,
            'scores': {
                'test_loss': {},
                'test_acc': {},
            },
            'capacity': {
                'ratio': [],
                'num_shared_activ': [],
                'num_adapts_activ': [],
            },
            'communication': {
                'ratio': [],
                'num_actives': [],
            },
            'num_total_params': 0,
            'optimizer': []
        }
        self.init_learning_rate()

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_train.npy'.format(cid))).item()
        self.optimizer = tf.keras.optimizers.deserialize(self.state['optimizer'])

    def save_state(self):
        self.state['optimizer'] = tf.keras.optimizers.serialize(self.optimizer)
        np_save(self.args.state_dir, '{}_train.npy'.format(self.state['client_id']), self.state)

    def init_learning_rate(self):
        self.state['early_stop'] = False
        self.state['lowest_lss'] = np.inf
        self.state['curr_lr'] = self.args.lr
        self.state['curr_lr_patience'] = self.args.lr_patience
        self.init_optimizer(self.state['curr_lr'])

    def init_optimizer(self, curr_lr):
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=curr_lr)

    def adaptive_lr_decay(self):
        vlss = self.vlss
        if vlss<self.state['lowest_lss']:
            self.state['lowest_lss'] = vlss
            self.state['curr_lr_patience'] = self.args.lr_patience
        else:
            self.state['curr_lr_patience']-=1
            if self.state['curr_lr_patience']<=0:
                prev = self.state['curr_lr']
                self.state['curr_lr']/=self.args.lr_factor
                self.logger.print(self.state['client_id'], 'epoch:%d, learning rate has been dropped from %.5f to %.5f' \
                                                    %(self.state['curr_epoch'], prev, self.state['curr_lr']))
                if self.state['curr_lr']<self.args.lr_min:
                    self.logger.print(self.state['client_id'], 'epoch:%d, early-stopped as minium lr reached to %.5f'%(self.state['curr_epoch'], self.state['curr_lr']))
                    self.state['early_stop'] = True
                self.state['curr_lr_patience'] = self.args.lr_patience
                self.optimizer.lr.assign(self.state['curr_lr'])

    def train_one_round(self, curr_round, round_cnt, curr_task, client_id = None):

        """
        Handles a single round of training
        """
        tf.keras.backend.set_learning_phase(1)
        self.state['curr_round'] = curr_round
        self.state['round_cnt'] = round_cnt
        self.state['curr_task'] = curr_task
        self.curr_model = self.nets.get_model_by_tid(client_id  = client_id)
        
        

        if self.args.base_network == "made":
            if curr_task > 0 and round_cnt == 0:
                atten = self.curr_model.get_weights("atten")["W_atten"][0].numpy()
                atten += self.curr_model.get_weights("atten")["W_atten"][1].numpy()
                atten += self.curr_model.get_weights("atten")["D_atten"][0].numpy()
                self.init_atten = atten/3
                print(np.round(atten/3-self.init_atten, 3))
            for epoch in range(self.args.num_epochs):
                self.state['curr_epoch'] = epoch+1
                batches = 0
                for i in range(0, len(self.task['x_train']), self.args.batch_size):
                    batches+=1
                    x_batch = tf.convert_to_tensor(self.task['x_train'][i:i+self.args.batch_size])
                    if False: #batches == len(self.task['x_train'])//self.args.batch_size and (curr_round == 0 or curr_round == self.args.num_rounds -1):
                        self.curr_model.train_step((x_batch, x_batch), lossf = self.params['loss'], extended_log = True) # in made case x_batch = y_batch
                    else:
                        self.curr_model.train_step((x_batch, x_batch), lossf = self.params['loss']) # in made case x_batch = y_batch
                self.validate(client_id)
                #self.adaptive_lr_decay()
                #if self.state['early_stop']:
                #   continue
            



    def validate(self, client_id):
        tf.keras.backend.set_learning_phase(0)
        if self.args.base_network == 'made':
            batches = 0
            loss = 0
            val_loss = 0
            #print(self.state['curr_task'])
            for i in range(0, len(self.task['x_valid']), self.args.batch_size):
                batches += 1
                x_batch = self.task['x_valid'][i:i+self.args.batch_size]
                y_batch = self.task['y_valid'][i:i+self.args.batch_size]
                y_pred = self.curr_model(x_batch)
                if batches == len(self.task['x_valid'])//self.args.batch_size and not self.args.only_federated:
                    loss += float(self.params['loss'](y_batch, y_pred, True))
                else:
                    loss += float(self.params['loss'](y_batch, y_pred))
                val_loss += float(self.params['val_loss'](y_batch, y_pred))
                self.add_performance('valid_lss', val_loss)
            print()
            #check_var= self.measure_performance('valid_lss')
            val_loss /= batches
            loss /= batches
            self.vlss= val_loss
            #print(f"Check_var is {check_var} and actual_var is {val_loss}")
            #print("Client_id is : ", client_id)
            if self.args.task == 'mnist':
                base_path= os.path.join(self.args.task_path, 'mnist')
            elif self.args.task == 'binary':
                base_path= os.path.join(self.args.task_path, 'binary')
            else:
                base_path= os.path.join(self.args.task_path, 'non_miid')
            #print(base_path)
            if self.state['curr_task'] >0:
                for j in range(self.state['curr_task'] ):
                    #print(os.path.join(base_path, f'NON_IID_{j*3+client_id}_test.npy'))
                    if self.args.task == 'mnist':
                        temporary_data= np.load(os.path.join(base_path, f'mnist_{j*self.args.num_clients +client_id}_test.npy'), allow_pickle=True).item()
                    elif self.args.task == 'binary':
                        temporary_data= np.load(os.path.join(base_path, f'binary_{j*self.args.num_clients +client_id}_test.npy'), allow_pickle=True).item()
                    else:
                        temporary_data= np.load(os.path.join(base_path, f'NON_IID_{j*self.args.num_clients +client_id}_test.npy'), allow_pickle=True).item()
                    
                    
                    valid= temporary_data['x_test']
                    batches = 0
                    t_loss = 0
                    t_val_loss = 0
                    #y_del = self.curr_model(valid)
                    #delt = float(self.params['val_loss'](y_del, valid))
                    #print("Temp loss", delt)
                    for i in range(0, len(valid), self.args.batch_size):
                        batches += 1
                        x_batch = valid[i:i+self.args.batch_size]
                        y_batch = valid[i:i+self.args.batch_size]
                        y_pred = self.curr_model(x_batch)
                        if batches == len(valid)//self.args.batch_size and not self.args.only_federated:
                            t_loss += float(self.params['loss'](y_batch, y_pred, True))
                        else:
                            t_loss += float(self.params['loss'](y_batch, y_pred))
                        t_val_loss += float(self.params['val_loss'](y_batch, y_pred))
                    t_val_loss /= batches
                    t_loss /= batches
                    self.logger.print(self.state['client_id'], 'forgetting for client: {} at task {} during curr_task {}: val_loss: {} '
                        .format(self.state['client_id'], j, self.state['curr_task'], t_val_loss)
                        )
            

        if self.args.base_network == 'made':
            if self.state['curr_epoch'] == self.args.num_epochs or self.args.only_federated:
                if self.state['round_cnt'] == self.args.num_rounds-1:
                    batches = 0
                    test_loss = 0
                    for i in range(0, len(self.task['x_test_list']), self.args.batch_size):
                        batches += 1
                        x_batch = self.task['x_test_list'][i:i+self.args.batch_size]
                        y_batch = self.task['y_test_list'][i:i+self.args.batch_size]
                        y_pred = self.curr_model(x_batch)
                        test_loss += float(self.params['loss'](y_batch, y_pred))
                    test_loss /= batches
                    self.logger.print(self.state['client_id'], 'round:{}(cnt:{}),epoch:{},task:{},test_lss:{} ({},#_train:{},#_valid:{},#_test:{})'
                        .format(self.state['curr_round'], self.state['round_cnt'], self.state['curr_epoch'], self.state['curr_task'], round(test_loss,3),
                        self.task['task_names'][self.state['curr_task']], len(self.task['x_train']), len(self.task['x_valid']),len(self.task['x_test_list'][self.state['curr_task']])))
                    if self.state['curr_task'] > 0 and self.state['round_cnt']==49:
                        atten = self.curr_model.get_weights("atten")["W_atten"][0].numpy()
                        atten += self.curr_model.get_weights("atten")["W_atten"][1].numpy()
                        atten += self.curr_model.get_weights("atten")["D_atten"][0].numpy()
                        print(np.round(atten/3-self.init_atten, 3))
                self.logger.print(self.state['client_id'], 'round:{}(cnt:{}),epoch:{},task:{},lss:{},val_lss:{} ({},#_train:{},#_valid:{},#_test:{})'
                    .format(self.state['curr_round'], self.state['round_cnt'], self.state['curr_epoch'], self.state['curr_task'], round(loss, 3), round(val_loss,3),
                    self.task['task_names'][self.state['curr_task']], len(self.task['x_train']), len(self.task['x_valid']),len(self.task['x_test_list'][self.state['curr_task']])))

    def evaluate(self):
        tf.keras.backend.set_learning_phase(0)
        for tid in range(self.state['curr_task']+1):
            if self.args.model == 'stl':
                if not tid == self.state['curr_task']:
                    continue
            x_test = self.task['x_test_list'][tid]
            y_test = self.task['y_test_list'][tid]
            model = self.nets.get_model_by_tid(tid)
            for i in range(0, len(x_test), self.args.batch_size):
                x_batch = x_test[i:i+self.args.batch_size]
                y_batch = y_test[i:i+self.args.batch_size]
                y_pred = model(x_batch)
                loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred)
                self.add_performance('test_lss', 'test_acc', loss, y_batch, y_pred)
            lss, acc = self.measure_performance('test_lss', 'test_acc')
            if not tid in self.state['scores']['test_loss']:
                self.state['scores']['test_loss'][tid] = []
                self.state['scores']['test_acc'][tid] = []
            self.state['scores']['test_loss'][tid].append(lss)
            self.state['scores']['test_acc'][tid].append(acc)
            self.logger.print(self.state['client_id'], 'round:{}(cnt:{}),epoch:{},task:{},lss:{},acc:{} ({},#_train:{},#_valid:{},#_test:{})'
                .format(self.state['curr_round'], self.state['round_cnt'], self.state['curr_epoch'], tid, round(lss, 3), \
                    round(acc, 3), self.task['task_names'][tid], len(self.task['x_train']), len(self.task['x_valid']), len(x_test)))

    def mnist_evaluate(self, test_set, cid):
        self.curr_model = self.nets.get_model_by_tid(client_id  = cid)
        x_test = test_set["x_test"]
        y_test = test_set["y_test"]
        y_pred = self.curr_model(x_test)
        loss = float(self.params['loss'](y_test, y_pred))
        self.logger.print(cid, 'final result on full test set: lss:{}, #_test_samples: {})'
            .format(round(loss, 9), len(x_test)))


    def add_performance(self, lss_name, loss):
        self.metrics[lss_name](loss)
        #self.metrics[acc_name](y_true, y_pred)

    def measure_performance(self, lss_name):
        lss = float(self.metrics[lss_name].result())
        #acc = float(self.metrics[acc_name].result())
        self.metrics[lss_name].reset_states()
        #self.metrics[acc_name].reset_states()
        return lss

    def calculate_capacity(self):
        def l1_pruning(weights, hyp):
            hard_threshold = np.greater(np.abs(weights), hyp).astype(np.float32)
            return weights*hard_threshold

        if self.state['num_total_params'] == 0:
            for dims in self.nets.shapes:
                params = 1
                for d in dims:
                    params *= d
                self.state['num_total_params'] += params
        num_total_activ = 0
        num_shared_activ = 0
        num_adapts_activ = 0
        for var_name in self.nets.decomposed_variables:
            if var_name == 'adaptive':
                for tid in range(self.state['curr_task']+1):
                    for lid in self.nets.decomposed_variables[var_name][tid]:
                        var = self.nets.decomposed_variables[var_name][tid][lid]
                        var = l1_pruning(var.numpy(), self.args.lambda_l1)
                        actives = np.not_equal(var, np.zeros_like(var)).astype(np.float32)
                        actives = np.sum(actives)
                        num_adapts_activ += actives
            elif var_name == 'shared':
                for var in self.nets.decomposed_variables[var_name]:
                    actives = np.not_equal(var.numpy(), np.zeros_like(var)).astype(np.float32)
                    actives = np.sum(actives)
                    num_shared_activ += actives
            else:
                continue
        num_total_activ += (num_adapts_activ + num_shared_activ)
        ratio = num_total_activ/self.state['num_total_params']
        self.state['capacity']['num_adapts_activ'].append(num_adapts_activ)
        self.state['capacity']['num_shared_activ'].append(num_shared_activ)
        self.state['capacity']['ratio'].append(ratio)
        self.logger.print(self.state['client_id'], 'model capacity: %.3f' %(ratio))

    def calculate_communication_costs(self, prams):
        if self.state['num_total_params'] == 0:
            for dims in self.nets.shapes:
                params = 1
                for d in dims:
                    params *= d
                self.state['num_total_params'] += params

        num_actives = 0
        for i, pruned in enumerate(prams):
            actives = np.not_equal(pruned, np.zeros_like(pruned)).astype(np.float32)
            actives = np.sum(actives)
            num_actives += actives

        ratio = num_actives/self.state['num_total_params']
        self.state['communication']['num_actives'].append(num_actives)
        self.state['communication']['ratio'].append(ratio)
        self.logger.print(self.state['client_id'], 'communication cost: %.3f' %(ratio))

    def set_details(self, details):
        self.params = details

    def set_task(self, task):
        self.task = task

    def get_scores(self):
        return self.state['scores']

    def get_capacity(self):
        return self.state['capacity']

    def get_communication(self):
        return self.state['communication']

    #updates[:][0] holds weights,
    #updates[:][1] holds training size
    def aggregate(self, updates):
        if self.args.base_network == 'made':
            if self.args.only_federated:
                client_weights = [u[0] for u in updates]
                new_weights = {"W": [], "bias": []}
                params = ["W", "bias"]
                [new_weights["W"].append(np.zeros(client_weights[0]["W"][l].shape)) for l in  range(len(client_weights[0]["W"]))]
                [new_weights["bias"].append(np.zeros(client_weights[0]["bias"][l].shape)) for l in  range(len(client_weights[0]["bias"]))]
                #new_weights["W"] = np.zeros(client_weights[0]["W"].shape)
                #new_weights["bias"] = np.zeros(client_weights[0]["bias"].shape)
                if self.args.connectivity_weights:
                    new_weights["U"] = []
                    params.append("U")
                    [new_weights["U"].append(np.zeros(client_weights[0]["U"][l].shape)) for l in  range(len(client_weights[0]["U"]))]
                if self.args.direct_input:
                    new_weights["D"] = []
                    params.append("D")
                    [new_weights["D"].append(np.zeros(client_weights[0]["D"][l].shape)) for l in  range(len(client_weights[0]["D"]))]
                    #new_weights["D"] = np.zeros(client_weights[0]["D"].shape)
                ratio = float(1 / len(client_weights))
                for weight in client_weights:
                    for param in params:
                        for count, layer in enumerate(weight[param]):
                            new_weights[param][count] += layer * ratio
                        #new_weights["W"] += np.array(weight["W"]) * ratio
                        #new_weights["bias"] += np.array(weight["bias"]) * ratio
                        #if self.args.connectivity_weights:
                    #        new_weights["U"] += np.array(weight["U"]) * ratio
                        #if self.args.direct_input != "disabled":
                        #    new_weights["D"] += np.array(weight["D"]) * ratio
                #temp = type(new_weights["W"][0][0])

            else:
                client_weights = [u[0][0] for u in updates]
                client_masks = [u[0][1] for u in updates] #binarized
                new_weights = {}
                epsi = 1e-15
                params = ["W"]
                if self.args.connectivity_weights:
                    params.append("U")
                if self.args.direct_input:
                    params.append("D")
                for param in params:
                    total_sizes = epsi
                    new_weights[f"{param}_global"] = []
                    [new_weights[f"{param}_global"].append(np.zeros(client_weights[0][f"{param}_global"][l].shape)) for l in  range(len(client_weights[0][f"{param}_global"]))]
                    old_weights = []
                    masks = []
                    for i in range(len(client_weights)):
                        if f"{param}_global" in client_weights[i]:
                            old_weights.append(client_weights[i][f"{param}_global"])
                        if f"{param}_mask" in client_masks[i]:
                            masks.append(client_masks[i][f"{param}_mask"])
                    masks = tf.ragged.constant(masks, dtype=tf.float32)
                    for mask in masks:
                        total_sizes += mask
                    for c_weights in old_weights: # by client
                        for lidx, l_weights in enumerate(c_weights): # by layer
                            ratio = 1/total_sizes[lidx]
                            #print(ratio)
                            new_weights[f"{param}_global"][lidx] += tf.math.multiply(l_weights, ratio).numpy()


                # weights have the form: List of updates (update = Dictionary with global variable parts: W_global, U_global... (Dictionary ENtry = List of weights (element in list = weights for one layer))
                #new_weights = {}
                #new_weights["W_global"] = []
                #params = ["W"]
                #[new_weights["W_global"].append(np.zeros(client_weights[0]["W_global"][l].shape)) for l in  range(len(client_weights[0]["W_global"]))]
                #epsi = 1e-15
                #total_sizes = epsi
                #client_masks = tf.ragged.constant(client_masks, dtype=tf.float32)
                #for mask in client_masks:
                #    total_sizes += mask
                #if self.args.connectivity_weights:
            #        new_weights["U_global"] = []
        #            params.append("U")
    #                [new_weights["U_global"].append(np.zeros(client_weights[0]["U_global"][l].shape)) for l in  range(len(client_weights[0]["U_global"]))]
#                for weight in client_weights:
#                    for param in params:
#                        for count, layer in enumerate(weight[param]):
#                            ratio = 1/total_sizes[lidx]
#                            new_weights[param][count] += layer * ratio
#                if self.args.direct_input != "disabled":
#                    new_weights["D_global"] = []
#                    [new_weights["D_global"].append(np.zeros(client_weights[0]["D_global"][l].shape)) for l in  range(len(client_weights[0]["D_global"]))]
#                    for weight in client_weights:
#                        for count, layer in enumerate(weight["D_global"]):
#                            new_weights["D_global"][count] += layer * ratio
        return new_weights
