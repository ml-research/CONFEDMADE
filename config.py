def set_config(args):

    args.output_path = '$path/to/outputfolder$'

    args.sparse_comm = True
    args.client_sparsity = 0.2
    args.server_sparsity = 0.2

    args.model ='fedweit'
    if args.task == 'non_iid_50':
        args.base_network = 'lenet'
    elif args.task == 'mnist':
        args.base_network = 'made'
    elif args.task == 'binary':
        args.base_network = 'made'
    elif args.task == 'non_miid':
        args.base_network = 'made'
    # adaptive learning rate
    args.lr_patience = 3
    args.lr_factor = 3
    args.lr_min = 1e-10

    # base network hyperparams
    if args.base_network == 'lenet':
        args.lr = 1e-3/3
        args.wd = 1e-4

    if args.base_network == 'made':
        args.lr = 1e-3 # adam learning rate
        args.wd = 1e-4

    if 'fedweit' in args.model:
        args.wd = 0
        args.lambda_l1 = 1e-4
        args.lambda_l2 = 100.
        args.lambda_mask = 0

    return args

def set_data_config(args):

    args.task_path = '$path/to/outputfolder$'

    args.ip_shape={
    

    'bbc':  1058,
    'c20ng':  910,
    'cr52': 889,
    #'cwebkb': 239,
    'moviereview':  1001,

    'tmovie':  500,
    'nips': 500,
    'book': 500,
    

    'rcv1': 150,
    'tretail': 135,
    'pumsb_star': 163,
    'dna': 180,
    'kosarek':  190,

    'kdd' : 65,
    'plants': 69,
    'baudio' : 100,
    'jester' : 100,
    'bnetflix' : 100,
    'accidents': 111,
    'mushrooms': 112, 
    'adult': 123,
    'connect4': 126,
    'ocr_letters': 128,
}

    # CIFAR10(0), CIFAR100(1), MNIST(2), SVHN(3),
    # F-MNIST((4), TrafficSign(5), FaceScrub(6), N-MNIST(7)

    if args.task in ['non_iid_50'] :
        args.datasets    = [0, 1, 2, 3, 4, 5, 6, 7]  # [0.193, 0.249, 0.363, 0.564], [0.184, 0.279, 0.566, 0.365], [0.22, 0.601, 0.275, 0.263], [0.515, 0.337, 0.283, 0.307]
        args.num_clients = 5
        args.num_tasks   = 10
        args.num_classes = 5
        args.frac_clients = 1.0
    
    elif args.task == 'binary':
        args.only_federated = False
        args.same_masks = True #Should clients use the same masks?
        args.same_input_order = True#for FedWeITMADE: should clients use same input ordering? DOnt use when order agnostic training is active
        args.datasets = [5]
        args.t_name=  ['adult', 'connect4', 'tretail', 'rcv1']#[ 'tretail','pumsb_star', 'dna', 'kosarek' ] #['baudio', 'jester', 'bnetflix', 'accidents', 'mushrooms', 'connect4'] ##['cwebkb','c20ng','cr52','moviereview','bbc']#['tmovie', 'nips', 'book'] ##
        args.input_size =  args.ip_shape[args.t_name[len(args.t_name)-1]]
        args.hidden_layers = [110] #[args.input_size//2] #hidden layer shapes
        args.task_name = "offline"
        args.natural_input_order = True
        args.num_clients = 1
        args.num_tasks   = 4
        args.num_classes = 4 
        args.frac_clients = 1.0
        args.num_masks = 2
        
        args.order_agn = True
        args.order_agn_step_size = 1
        args.conn_agn_step_size = 1
        args.connectivity_weights = True
        args.apply_madeloss= True
        args.apply_mademask= True
        args.direct_input = True
        args.experiment =  "incre_lb" #"label permutation across clients" #no_overlap_distinct_domain_task" #"No_Overlap_with_Mixed_label_task"#"else"    #"other"  


    elif args.task == 'mnist':
        args.only_federated = False
        args.same_masks = True #Should clients use the same masks?
        args.same_input_order = False #for FedWeITMADE: should clients use same input ordering? DOnt use when order agnostic training is a ctive
        args.datasets = [2]
        args.input_size= 784
        args.hidden_layers = [400] #hidden layer shapes
        args.mnist_path = '/content/drive/MyDrive/binarized_mnist.npz'
        args.natural_input_order = False
        args.num_clients = 3
        args.num_tasks   = 5
        args.num_classes = 1
        args.frac_clients = 1.0
        args.num_masks = 1
        args.order_agn = True
        args.order_agn_step_size = 1
        args.conn_agn_step_size = 1
        args.connectivity_weights = False
        args.direct_input = True
        args.experiment = "incremenntal_lowerbound"

    elif args.task == 'non_miid':
        args.only_federated = False
        args.same_masks = True #Should clients use the same masks?
        args.same_input_order = True #for FedWeITMADE: should clients use same input ordering? DOnt use when order agnostic training is active
        args.datasets = [6]
        args.input_size = 784
        args.hidden_layers = [500] #hidden layer shapes
        args.mnist_path = '/content/drive/MyDrive/binarized_mnist.npz'
        args.natural_input_order = True
        args.num_clients = 3
        args.num_tasks   = 3
        args.num_classes = 3
        args.frac_clients = 1.0
        args.task_name = "offline"
        args.num_masks = 1
        args.order_agn = False
        args.order_agn_step_size = 1
        args.conn_agn_step_size = 1
        args.connectivity_weights = True
        args.apply_madeloss= True
        args.apply_mademask= True
        args.direct_input = True
        args.experiment =   " " #"no_overlap_distinct_domain_task" #"No_Overlap_with_Mixed_label_task"#"else"    #"other"  

    else:
        print('no correct task was given: {}'.format(args.task))

    return  args
