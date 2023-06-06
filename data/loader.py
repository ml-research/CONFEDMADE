import os
import pdb
import glob
import numpy as np

from misc.utils import *

class DataLoader:
    """ Data Loader

    Loading data for the corresponding clients

    """
    def __init__(self, args):
        self.args = args
        self.base_dir = os.path.join(self.args.task_path, self.args.task)
        self.did_to_dname = {
            0: 'cifar10',
            1: 'cifar100',
            2: 'mnist',
            3: 'svhn',
            4: 'fashion_mnist',
            5: 'binary',
            6: 'face_scrub',
            7: 'not_mnist',
        }

    def init_state(self, cid):
        self.state = {
            'client_id': cid,
            'tasks': []
        }
        self.load_tasks(cid)

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_data.npy'.format(cid))).item()

    def save_state(self):
        np_save(self.args.state_dir, '{}_data'.format(self.state['client_id']), self.state)

    def load_tasks(self, cid):

        if self.args.task == 'mnist':
            if self.args.only_federated:
                task_set = {}
                for c in range(self.args.num_clients):
                    client_tasks = [f'mnist_{c}']
                    task_set[c] = client_tasks
                self.state['tasks'] = task_set[self.state['client_id']]
            else:
                task_set = {}
                for c in range(self.args.num_clients):
                    client_tasks = []
                    for t in range(self.args.num_tasks):
                        client_tasks.append('mnist_{}'.format(c+t*self.args.num_clients))
                        #client_tasks.append('mnist_{}'.format(c))
                    task_set[c] = client_tasks
                self.state['tasks'] = task_set[self.state['client_id']]
        elif self.args.task == 'non_miid':
            if self.args.only_federated:
                task_set = {}
                for c in range(self.args.num_clients):
                    client_tasks = [f'mnist_{c}']
                    task_set[c] = client_tasks
                self.state['tasks'] = task_set[self.state['client_id']]
            else:
                task_set = {}
                for c in range(self.args.num_clients):
                    client_tasks = []
                    for t in range(self.args.num_tasks):
                        client_tasks.append('NON_IID_{}'.format(c+t*self.args.num_clients))
                        #client_tasks.append('mnist_{}'.format(c))
                    task_set[c] = client_tasks
                self.state['tasks'] = task_set[self.state['client_id']]

        elif self.args.task == 'binary':
            if self.args.only_federated:
                task_set = {}
                for c in range(self.args.num_clients):
                    client_tasks = [f'binary_{c}']
                    task_set[c] = client_tasks
                self.state['tasks'] = task_set[self.state['client_id']]
            else:
                task_set = {}
                for c in range(self.args.num_clients):
                    client_tasks = []
                    for t in range(self.args.num_tasks):
                        client_tasks.append('binary_{}'.format(c+t*self.args.num_clients))
                        #client_tasks.append('mnist_{}'.format(c))
                    task_set[c] = client_tasks
                self.state['tasks'] = task_set[self.state['client_id']]
        else:
            print('no correct task was given: {}'.format(self.args.task))
            os._exit(0)

    def get_train(self, task_id):
        return load_task(self.base_dir, self.state['tasks'][task_id]+'_train.npy').item()

    def get_full_test(self):
        return load_task(self.base_dir, 'mnist_full_test_set.npy').item()

    def get_valid(self, task_id):
        valid = load_task(self.base_dir, self.state['tasks'][task_id]+'_valid.npy').item()
        return valid['x_valid'], valid['y_valid']

    def get_test(self, task_id):
        if self.args.base_network == "made":
            test = load_task(self.base_dir, self.state['tasks'][task_id]+'_test.npy').item()
            return test['x_test'], test['y_test']
        x_test_list = []
        y_test_list = []
        for tid, task in enumerate(self.state['tasks']):
            if tid <= task_id:
                test = load_task(self.base_dir, task+'_test.npy').item()
                x_test_list.append(test['x_test'])
                y_test_list.append(test['y_test'])
        return x_test_list, y_test_list
