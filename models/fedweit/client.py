import pdb
import math
import random
import tensorflow as tf
from keras import backend as k
from keras import metrics
from misc.utils import *
from modules.federated import ClientModule

class Client(ClientModule):
    
    """ CONFEDMADE Client: Performing cleint training, saving them """

    def __init__(self, gid, args, initial_weights, cid_per_gpu):
        self.cid_per_gpu = cid_per_gpu
        super(Client, self).__init__(gid, args, initial_weights)
        self.state['gpu_id'] = gid

    def train_one_round(self, client_id, curr_round, selected, global_weights=None, from_kb=None):
        ######################################
        self.switch_state(client_id)
        ######################################
        self.state['round_cnt'] += 1
        self.state['curr_round'] = curr_round
        self.client= client_id
        if not from_kb == None:
            if self.args.base_network == "made" and self.args.only_federated == False:
                self.nets.set_weights(from_kb, client_id, type = "kb")
            else:
                for lid, weights in enumerate(from_kb):
                    tid = self.state['curr_task']+1
                    self.nets.decomposed_variables['from_kb'][tid][lid].assign(weights)
        if self.state['curr_task']<0:
            self.init_new_task(client_id)
            #set weights global, kb = []
            self.set_weights(global_weights, client_id, type = "global")
        else:
            is_last_task = (self.state['curr_task']==self.args.num_tasks-1)
            is_last_round = (self.state['round_cnt']%self.args.num_rounds==0 and self.state['round_cnt']!=0)
            is_last = is_last_task and is_last_round
            if is_last_round:
                if is_last_task:
                    if self.train.state['early_stop']:
                        self.train.evaluate()
                    self.stop()
                    return
                else:
                    self.init_new_task(client_id)
                    if self.args.base_network != 'made':
                        self.state['prev_body_weights'] = self.nets.get_body_weights(self.state['curr_task'])
                        #self.state['prev_weights'] = self.nets.get_weights(self.state['curr_task'], client_id)
            else:
                self.load_data()

        if selected:
            self.set_weights(global_weights, client_id, type = "global")

        with tf.device('/device:GPU:{}'.format(self.state['gpu_id'])):
            self.train.train_one_round(self.state['curr_round'], self.state['round_cnt'], self.state['curr_task'], client_id)
        if not self.args.only_federated:
            self.logger.save_current_state(self.state['client_id'], {
                'scores': self.train.get_scores(),
                'capacity': self.train.get_capacity(),
                'communication': self.train.get_communication()
                })
        self.save_state()

        if selected:
            if self.args.base_network == "made" and not self.args.only_federated:
                return self.get_weights(client_id, type = "to_server"), self.get_train_size()
            else:
                return self.get_weights(client_id), self.get_train_size()

    def mnist_eval_round(self, cid, weights):
        self.switch_state(cid)
        self.set_weights(weights, cid, type = "global")
        with tf.device('/device:GPU:{}'.format(self.state['gpu_id'])):
            self.train.mnist_evaluate(self.loader.get_full_test(), cid)

    def cross_entropy_loss(self, x, x_decoded_mean):
        x = k.flatten(x)
        x_decoded_mean = k.flatten(x_decoded_mean)
        xent_loss = self.args.input_size * metrics.binary_crossentropy(x, x_decoded_mean) #transform average cross entropy loss to absolute loss
        return xent_loss

    def made_fedweit_loss(self, x, x_decoded_mean, extended_log = False):
        weight_decay, sparseness, sparseness_log, approx_loss = 0, 0, 0, 0
        #get weights needed to calculate loss
        sw = self.nets.get_weights(self.state['client_id'], "global")
        sw_last_task = self.nets.get_weights(self.state['client_id'], "global_last_task")
        made_mask= self.nets.get_MADE_mask()
        mask = self.nets.get_weights(self.state['client_id'], "mask")
        prev_masks = self.nets.get_weights(self.state['client_id'], "prev_masks")
        aws = self.nets.get_weights(self.state['client_id'], "all_adapts")
        adapts_last_task = self.nets.get_weights(self.state['client_id'], "adapts_last_task")

        weights = ["W"]
        if self.args.connectivity_weights:
            weights.append("U")
        if self.args.direct_input:
            weights.append("D")

        # Cross entropy Loss
        loss = self.cross_entropy_loss(x, x_decoded_mean)

        for weight in weights:
            for lid in range(len(self.nets.shapes)):
                if weight == "D" and lid >= len(sw["D_global"]):
                    continue  #since D is not part of each layer

                #Weight decay if enabled
                weight_decay += self.args.wd * tf.nn.l2_loss(aws[f"{weight}_all_adapts"][lid][self.state["curr_task"]])
                weight_decay += self.args.wd * tf.nn.l2_loss(mask[f"{weight}_mask"][lid])
                made_mask_layer= made_mask[str(self.client)][lid]
                
                # task independent Sparsity loss
                if weight == "D":
                    sparseness += self.args.lambda_l1 * tf.reduce_sum(tf.abs(aws[f"{weight}_all_adapts"][lid][self.state["curr_task"]]))
                    sparseness_log += tf.reduce_sum(tf.abs(aws[f"{weight}_all_adapts"][lid][self.state["curr_task"]]))
                    sparseness += self.args.lambda_mask * tf.reduce_sum(tf.abs(mask[f"{weight}_mask"][lid]))
                else:
                    sparseness += self.args.lambda_l1 *  tf.reduce_sum(tf.abs(aws[f"{weight}_all_adapts"][lid][self.state["curr_task"]] * made_mask_layer))
                    sparseness_log += tf.reduce_sum(tf.abs(aws[f"{weight}_all_adapts"][lid][self.state["curr_task"]]))
                    sparseness += self.args.lambda_mask * tf.reduce_sum(tf.abs(mask[f"{weight}_mask"][lid]))


                if self.state['curr_task'] == 0:
                    weight_decay += self.args.wd * tf.nn.l2_loss(sw[f"{weight}_global"][lid])
                else:
                    sw_delta = sw[f"{weight}_global"][lid]-sw_last_task[f"{weight}_global_last_task"][lid]
                    for tid in range(self.state['curr_task']):
                        prev_mask = prev_masks[f"{weight}_prev_masks"][lid][tid]
                        prev_aw = aws[f"{weight}_all_adapts"][lid][tid]
                        #################################################

                        # catastrophic forgetting loss
                        adapt_delta = prev_aw - adapts_last_task[f"{weight}_adapts_last_task"][lid][tid]
                        if weight == "D":
                            a_l2 = tf.nn.l2_loss(sw_delta  * prev_mask + adapt_delta)
                        else:
                            a_l2 = tf.nn.l2_loss(sw_delta   * prev_mask  + adapt_delta  )
                    
                        #a_l2 = tf.nn.l2_loss(sw_delta * prev_mask + adapt_delta)
                        approx_loss += self.args.lambda_l2 * a_l2
                        #################################################

                        # sparseness task adaptive part

                        #sparseness += self.args.lambda_l1 * tf.reduce_sum(tf.abs(prev_aw))
                        if weight == "D":
                            sparseness += self.args.lambda_l1 * tf.reduce_sum(tf.abs(prev_aw))
                        else:    
                            sparseness += self.args.lambda_l1 * tf.reduce_sum(tf.abs(prev_aw)* made_mask_layer)
                    
                        sparseness_log += tf.reduce_sum(tf.abs(prev_aw))


        loss += weight_decay + sparseness + approx_loss
        return loss

    
    def get_adaptives(self, client_id = None):
        if self.args.base_network == "made":
            adapts = {}
            aw = self.nets.get_weights(client_id, type = "adaptive")
            for key in aw:
                adapts[f"{key}"] = []
                for lid in range(len(self.nets.shapes)):
                    if key == "D_adaptive":
                        if lid == 0:
                            hard_threshold = np.greater(np.abs(aw[f"{key}"][lid]), self.args.lambda_l1).astype(np.float32)
                            adapts[f"{key}"].append(aw[f"{key}"][lid] * hard_threshold)
                        else:
                            continue
                    else:
                        hard_threshold = np.greater(np.abs(aw[f"{key}"][lid]), self.args.lambda_l1).astype(np.float32)
                        adapts[f"{key}"].append(aw[f"{key}"][lid] * hard_threshold)

        return adapts
