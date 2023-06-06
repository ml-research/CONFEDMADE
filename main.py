import os
#from rtpt import RTPT
from prser import Parser
from datetime import datetime
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.experimental.set_memory_growth(physical_devices[1], True)
#tf.config.experimental.set_memory_growth(physical_devices[2], True)
#tf.config.experimental.set_memory_growth(physical_devices[3], True)
#tf.config.experimental.set_memory_growth(physical_devices[4], True)
#tf.config.experimental.set_memory_growth(physical_devices[5], True)
#tf.config.experimental.set_memory_growth(physical_devices[6], True)
#tf.config.experimental.set_memory_growth(physical_devices[7], True)
import os 
#os.environ['CUDA_VISIBLE_DEVICES']="0,1,2"
from config import *
from misc.utils import *
from data.generator import DataGenerator

def main(args):

    args = set_data_config(args)
    if args.work_type == 'gen_data':
        dm = DataGenerator(args)
        #dm.generate_data()
    else:
        args = set_config(args)
        os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
        
        now = datetime.now().strftime("%Y%m%d-%H%M")
        args.log_dir = os.path.join(args.output_path, 'logs/{}-{}-{}'.format(now, args.model, args.task))
        args.state_dir = os.path.join(args.output_path, 'states/{}-{}-{}'.format(now, args.model, args.task))
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.isdir(args.state_dir):
            os.makedirs(args.state_dir)
        
        if args.model == 'fedweit':
            from models.fedweit.server import Server
            server = Server(args)
            server.run()
        
        else:
            print('incorrect model was given: {}'.format(args.model))
            os._exit(0)

if __name__ == '__main__':
    #rtpt= RTPT(name_initials= 'ml_spaul', experiment_name= 'Training_Federated', max_iterations= 1000)
    #rtpt.start()
    main(Parser().parse())
