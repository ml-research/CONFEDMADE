import pdb
import sys
import time
import random
import threading
import tensorflow as tf

from misc.utils import *
from .client import Client
from modules.federated import ServerModule

class Server(ServerModule):
    """ CONFEDMADE Server """
    
    def __init__(self, args):
        super(Server, self).__init__(args, Client)
        self.client_adapts = [] #always only save client adaptives from last round

    def train_clients(self):
        cids = np.arange(self.args.num_clients).tolist()
        num_selection = int(round(self.args.num_clients*self.args.frac_clients))
        for curr_round in range(self.args.num_rounds*self.args.num_tasks):
            self.updates = []
            self.curr_round = curr_round+1
            self.is_last_round = self.curr_round%self.args.num_rounds==0
            if self.is_last_round:
                self.client_adapts = []
            selected_ids = random.sample(cids, num_selection) # pick clients
            self.logger.print('server', 'round:{} train clients (selected_ids: {})'.format(curr_round, selected_ids))
            self.logger.print('server', f'Length of parallel_clients = {len(self.parallel_clients)}')
            # train selected clients in parallel
            for clients in self.parallel_clients:
                self.threads = []
                for gid, cid in enumerate(clients):
                    client = self.clients[gid]
                    selected = True if cid in selected_ids else False
                    with tf.device('/device:GPU:{}'.format(gid)):
                        thrd = threading.Thread(target=self.invoke_client, args=(client, cid, curr_round, selected, self.get_weights(), self.get_adapts()))
                        self.threads.append(thrd)
                        thrd.start()
                # wait all threads each round
                for thrd in self.threads:
                    thrd.join()
            # update
            
            aggr = self.train.aggregate(self.updates)
            self.set_weights(aggr)
        if self.args.only_federated:
            for clients in self.parallel_clients:
                for gid, cid in enumerate(clients):
                    thrd = threading.Thread(target=self.mnist_test_run, args=(client, cid, self.get_weights()))
                    self.threads.append(thrd)
                    thrd.start()
            for thrd in self.threads:
                thrd.join()
        self.logger.print('server', 'done. ({}s)'.format(time.time()-self.start_time))
        sys.exit()

    def invoke_client(self, client, cid, curr_round, selected, weights, adapts):
        update = client.train_one_round(cid, curr_round, selected, weights, adapts)
        if not update == None:
            self.updates.append(update) # get global weights
            if self.is_last_round:
                if not self.args.only_federated:
                    self.client_adapts.append(client.get_adaptives(cid)) #build knowledge base

    def mnist_test_run(self, client, cid, weights):
        client.mnist_eval_round(cid, weights)

    def get_adapts(self):
        if self.args.only_federated:
            return
        if self.curr_round%self.args.num_rounds==1 and not self.curr_round==1:
            if self.args.base_network == "made":
                #returns: Dict with keys W/U/D_kb containing lists of weights
                #self.client_adapts contains List of
                from_kb = {}
                for key in self.client_adapts[0]:
                    from_kb[key] = []
                    for adapt_weight in self.client_adapts:
                        from_kb[key].append(adapt_weight[key])
                return from_kb
        else:
            return None
