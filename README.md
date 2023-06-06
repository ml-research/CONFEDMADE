
# Masked Autoencoders are Efficient Continual Federated Learners

This repository is an official Tensorflow 2 implementation of [Masked Autoencoders are Efficient Continual Federated Learners]



The main contributions of this work are as follows:

* We draw inspiration from the supervised FedWeIT and extend it to our unsupervised **Con**tinual **Fed**erated **MA**sked autoencoders for **D**ensity **E**stimation (**CONFEDMADE**); an unsupervised continual federated learner based on masking to enable selective knowledge transfer between clients and reduce forgetting.* Through our intelligent masking strategy, we are still successful in achieving desirable performances even after sparsifying the model parameters by 70 %. 
* We highlight that MADE is a model particularly amenable to CFL and investigate several non-trivial considerations, such as connectivity and masking strategy, beyond a trivial application of federated averaging and FedWeIT to the unsupervised setting.
* We extensively evaluate our approach on several CFL scenarios on both image and numerical data. Overall, CONFEDMADE consistently reduces forgetting while sparsifying parameters and reducing communication costs with respect to a naive unsupervised CFL approach.

## Credits 
* We have implemented some segments of our continual federated framework using the official repository of [FedWeIT](https://github.com/wyjeong/FedWeIT/tree/main). 
* Our tensorflow implentation of MADE is inspired from the Official Theano implementation of [MADE](https://github.com/mgermain/MADE/tree/master). 
## Environmental Setup

Please install packages from `requirements.txt` after creating your own environment with `python 3.8.x`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

Please specify your custom path to store the generated task sets for all the clients in FL and also output files.
```python
args.task_path = '/path/to/taskset/'  # for task sets of each client
args.output_path = '/path/to/outputs/' # for logs, weights, etc.
```
Also, Please specify the path to log your file in the misc/logger.py. 

## Data Generation

Run below script to generate datasets. 
The --task parameter has three choices: `mnist`, `bianry`, and `non_miid (Mnist+ Emnist)` to generate the desired type of task sets for the clients.

```bash
python3 ../main.py --work-type gen_data --task mnist --seed 777 
```  

## Run Experiments
To reproduce experiments, please execute `train_mnist.sh` file, or you may run the following comamnd line directly:

```bash
python3 ../main.py --gpu 0,1,2 \
		--work-type train \
		--model fedweit \
		--task mnist \
	 	--gpu-mem-multiplier 9 \
		--num-rounds 20 \
		--num-epochs 1 \
		--batch-size 100 \
		--seed 777 
```
Please refer to the config.py file to explore the other possible options (i.e hyperparameters, etc ). You can define any number of gpus with the following --gpu argument. Depending on the number of available gpus, the participating clients are logically switches across them


## Results
All clients and server save their evaluation metrics such as training and validation loss at the `\path\to\logfile` in the logger.py. Additionally, they create their own log files in `\path\to\output\logs\`, which include the experimental setups, such as learning rate, batch-size, etc. The log files will be updated for every comunication rounds. 


