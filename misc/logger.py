from datetime import datetime

from misc.utils import *

class Logger:
    """ Logging Module """

    def __init__(self, args, client_id=None):
        self.args = args
        self.options = vars(self.args)

    def print(self, client_id, message):
        name = 'server' if client_id == 'server' else f'client-{client_id}' 
        print(f'[{datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}]'+
                f'[{self.args.model}]'+
                f'[{self.args.task}]'+
                f'[{name}] '+
                f'{message}')
        f = open("output/Exp_order_agn_true_wo.txt", "a")   ### specify your path here to log the outputs"""
        f.write('\n'+f'[{datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}]'+
                f'[{self.args.model}]'+
                f'[{self.args.task}]'+
                f'[{name}] '+
                f'{message}')
        f.close()

    def save_current_state(self, client_id, current_state):
        current_state['options'] = self.options
        name = 'server' if client_id == 'server' else f'client-{client_id}' 
        write_file(self.args.log_dir, f'{name}.txt', current_state)
