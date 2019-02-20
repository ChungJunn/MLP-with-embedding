'''
test.py
-------
set up model with pre-trained options and parameter and evaluate it on testset data

command line arguments using argparse
-------------------------------------
--data_prefix : str (path)
name of dataset

--isValidation : boolean
whether model will be tested on validation set or test set

--batch_size : int
number of samples in a batch
'''

import argparse
parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--data_prefix", type=str, default='unsw', help="output dir")
parser.add_argument("--loadfrom", type=str, default='', help="output dir")
parser.add_argument("--batch_size", type=int, default=128, help="output dir")
args = parser.parse_args()

'''
codes adopted from https://pytorch.org tutorial 60 minutes blitz
'''
# load model options
print("loading model_options...")
import _pickle as pkl
with open(args.data_prefix + "." + args.loadfrom + ".model_options.pkl", "rb") as fp:
	model_options = pkl.load(fp)
with open(args.data_prefix + ".cols.pkl", "rb") as fp:
	cols_dict = pkl.load(fp)

cat_dims = cols_dict["cat_dims"]
emb_dims = [(cat_dim, min(10, cat_dim // 2)) for cat_dim in cat_dims]
no_of_cont = len(cols_dict["cont_cols"])
lin_layer_sizes = [model_options['hidden1'], model_options['hidden2'], model_options['hidden3'], model_options['hidden4']]
num_hidden = model_options['num_hidden']
lin_layer_sizes = lin_layer_sizes[:num_hidden]

#build model
import torch
import torch.cuda
from model import FeedForwardNN

print("building model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedForwardNN(emb_dims, no_of_cont=no_of_cont, lin_layer_sizes=lin_layer_sizes, output_size=model_options['output_dim']).to(device)

# setup dataset
import pandas as pd
from model import TabularDataset
from torch.utils.data import DataLoader

print("importing dataset from pkl file...")
df_train = pd.read_pickle(args.data_prefix + ".train.pkl")
df_test = pd.read_pickle(args.data_prefix + ".test.pkl")


testset = TabularDataset(df_test, cat_cols=cols_dict["cat_cols"], output_col=cols_dict["output_cols"])
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=1)

# test
model.test(testloader=testloader,
	data_prefix=args.data_prefix,
	loadfrom=args.loadfrom, 
	device=device)


