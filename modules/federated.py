import os
import sys
import pdb
import copy
import time
import math
import random
import threading
import atexit
import tensorflow as tf

from misc.utils import *
from misc.logger import Logger
from data.loader import DataLoader
from modules.nets import NetModule
from modules.train import TrainModule

class ServerModule:
    """ Serves as a Superclass for Server Module
    Handles the main run function, initializing global weights, assigning gpu memory, saving and loading weights.
    """
    def __init__(self, args, ClientObj):
        self.args = args
        self.clients = {}
        self.threads = []
        self.ClientObj = ClientObj
        self.limit_gpu_memory()
        self.logger = Logger(self.args)
        self.nets = NetModule(self.args)
        self.train = TrainModule(self.args, self.logger, self.nets)

        self.nets.init_state(None)
        self.train.init_state(None)
        atexit.register(self.atexit)

    def limit_gpu_memory(self):
        self.gpu_ids = np.arange(len(self.args.gpu.split(','))).tolist()
        self.gpus = tf.config.list_physical_devices('GPU')
        if len(self.gpus)>0:
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu = self.gpus[gpu_id]
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*self.args.gpu_mem_multiplier)])

    def run(self):
        self.logger.print('server', 'started')
        self.start_time = time.time()
        self.init_global_weights()
        self.init_clients()
        self.train_clients()

    def init_global_weights(self):
        self.global_weights = self.nets.init_global_weights()

    def init_clients(self):
        opt_copied = copy.deepcopy(self.args)
        num_gpus = len(self.gpu_ids)
        num_iteration = self.args.num_clients//num_gpus
        residuals = self.args.num_clients%num_gpus
        cid_per_gpu = []
        [cid_per_gpu.append([]) for i in range(num_gpus)]

        offset = 0
        self.parallel_clients = []
        for i in range(num_iteration):
            offset = i*num_gpus
            self.parallel_clients.append(np.arange(num_gpus)+offset)
            for gid in range(num_gpus):
                cid_per_gpu[gid].append(offset+gid)

        if residuals>0:
            offset = self.parallel_clients[-1][-1]+1
            self.parallel_clients.append(np.arange(residuals)+offset)

        initial_weights = self.global_weights
        if len(self.gpus)>0:
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu = self.gpus[gpu_id]
                with tf.device('/device:GPU:{}'.format(gpu_id)):
                    self.clients[gpu_id] =self.ClientObj(gpu_id, opt_copied, initial_weights, cid_per_gpu[i])
        else:
            num_parallel = 1
            self.clients = {i:self.ClientObj(i, opt_copied, initial_weights, cid_per_gpu[i]) for i in range(num_parallel)}

    def get_weights(self):
        return self.global_weights

    def set_weights(self, weights):
        self.global_weights = weights

    def atexit(self):
        for thrd in self.threads:
            thrd.join()
        self.logger.print('server', 'all client threads have been destroyed.' )


class ClientModule:
    """ Serves as a Superclass for Client Module
    Handles functionalities such as initializing a new client, switching between clients,
    setting the weights before being communicated to the Server.

    """
    def __init__(self, gid, args, initial_weights):
        self.args = args
        self.state = {'gpu_id': gid}
        self.lock = threading.Lock()
        self.logger = Logger(self.args)
        self.loader = DataLoader(self.args)
        self.init_weights = None
        self.update_weights = None
        self.nets = NetModule(self.args)
        self.train = TrainModule(self.args, self.logger, self.nets)
        self.made_mask= self.nets.get_MADE_mask()
        self.init_model(initial_weights)

    def init_model(self, initial_weights):
        decomposed = True if self.args.model in ['fedweit'] else False

        if self.args.base_network == 'made':
            self.nets.build_made(initial_weights, self.cid_per_gpu, decomposed=decomposed)

    def switch_state(self, client_id):
        if self.is_new(client_id):
            self.loader.init_state(client_id)
            self.nets.init_state(client_id)
            self.train.init_state(client_id)
            self.init_state(client_id)
        else: # load_state
            self.load_state(client_id)
            self.loader.load_state(client_id)
            self.nets.load_state(client_id)
            self.train.load_state(client_id)

    def is_new(self, client_id):
        return not os.path.exists(os.path.join(self.args.state_dir, f'{client_id}_client.npy'))

    def init_state(self, cid):
        self.state['client_id'] = cid
        self.state['task_names'] = {}
        self.state['curr_task'] =  -1
        self.state['round_cnt'] = 0
        self.state['done'] = False

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_client.npy'.format(cid))).item()
        self.update_train_config_by_tid(self.state['curr_task'], cid)

    def save_state(self):
        np_save(self.args.state_dir, '{}_client.npy'.format(self.state['client_id']), self.state)
        self.loader.save_state()
        self.nets.save_state()
        self.train.save_state()

    def init_new_task(self, client_id):
        self.state['curr_task'] += 1
        self.state['round_cnt'] = 0
        self.load_data()
        self.train.init_learning_rate()
        self.update_train_config_by_tid(self.state['curr_task'], client_id)

    def update_train_config_by_tid(self, tid, client_id = None):
        self.target_model = self.nets.get_model_by_tid(tid, client_id) if self.args.base_network != "made" else None
        self.trainable_variables = self.nets.get_trainable_variables(tid, client_id) if self.args.base_network != "made" else None
        self.trainable_variables_body = self.nets.get_trainable_variables(tid, client_id, head=False) if self.args.base_network != "made" else None
        if self.args.base_network == "made":
            loss = self.cross_entropy_loss if self.args.only_federated else self.made_fedweit_loss


        self.train.set_details({
            'loss': loss,
            'val_loss': self.cross_entropy_loss,
            'model': self.target_model,
            'trainables': self.trainable_variables,
        })

    def load_data(self):
        data = self.loader.get_train(self.state['curr_task'])
        self.state['task_names'][self.state['curr_task']] = data['name']
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_valid, self.y_valid = self.loader.get_valid(self.state['curr_task'])
        self.x_test_list, self.y_test_list = self.loader.get_test(self.state['curr_task'])
        self.train.set_task({
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_valid': self.x_valid,
            'y_valid': self.y_valid,
            'x_test_list': self.x_test_list,
            'y_test_list': self.y_test_list,
            'task_names': self.state['task_names'],
        })
    
    def get_model_by_tid(self, tid):
        return self.nets.get_model_by_tid(tid)

    def set_weights(self, weights, client_id = None, type = "global"):
        if self.args.model in ['fedweit']:
            if self.args.base_network == 'made':
                if weights is None:
                    return None
                self.nets.set_weights(weights, client_id, type)
            else:
                if weights is None:
                    return None
                for i, w in enumerate(weights):
                    sw = self.nets.get_variable('shared', i)
                    residuals = tf.cast(tf.equal(w, tf.zeros_like(w)), dtype=tf.float32)
                    sw.assign(sw*residuals+w)
        else:
            self.nets.set_body_weights(weights)

    def get_weights(self, client_id = None, type = None):
        if self.args.model in ['fedweit']:
            if self.args.base_network == 'made':
                if type == "to_server":
                    if self.args.model in ['fedweit']:
                        if self.args.sparse_comm:
                            params = ["W"]
                            if self.args.connectivity_weights:
                                params.append("U")
                            if self.args.direct_input:
                                params.append("D")
                            masks = self.nets.get_weights(client_id, type="mask")
                            global_weights = self.nets.get_weights(client_id, type="global")
                            sw_pruned = {}
                            hard_threshold = {}
                            for param in params:
                                hard_threshold[f"{param}_mask"] = []
                                sw_pruned[f"{param}_global"] = []
                                for lid, sw in enumerate(global_weights[f"{param}_global"]):
                                    mask = masks[f"{param}_mask"][lid]
                                    made_mask= self.made_mask[str(client_id)][lid]
                                    m_sorted = tf.sort(tf.keras.backend.flatten(tf.abs(mask)))
                                    thres = m_sorted[math.floor(len(m_sorted)*(self.args.client_sparsity))]
                                    m_bianary = tf.cast(tf.greater(tf.abs(mask), thres), tf.float32).numpy().tolist()
                                    hard_threshold[f"{param}_mask"].append(m_bianary)
                                    if param == 'D':
                                        sw_pruned[f"{param}_global"].append(sw.numpy() * m_bianary)
                                    else:
                                        if self.args.task == 'mnist' or self.args.task == 'non_miid':
                                            sw_pruned[f"{param}_global"].append(sw.numpy() * made_mask * m_bianary )
                                        else:
                                            sw_pruned[f"{param}_global"].append(sw.numpy() * m_bianary )
                                        #sw_pruned[f"{param}_global"].append(sw.numpy()) #.numpy()) # * m_bianary)
                                #sw_pruned[f"{param}_global"]=  global_weights[f"{param}_global"]   
                                    #sw_pruned[f"{param}_global"].append(sw.numpy()*m_bianary)
                                self.train.calculate_communication_costs(sw_pruned[f"{param}_global"])
                            return sw_pruned, hard_threshold
                else:
                    return self.nets.get_weights(client_id, type)

        else:
            return self.nets.get_body_weights()

    def get_train_size(self):
        return len(self.x_train)

    def get_task_id(self):
        return self.curr_task

    def stop(self):
        self.done = True
