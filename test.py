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
parser.add_argument("--isValidation", type=bool, default=False, help="output dir")
parser.add_argument("--batch_size", type=int, default=128, help="output dir")

args = parser.parse_args()

'''
codes adopted from https://pytorch.org tutorial 60 minutes blitz
'''
# load model options
print("loading model_options...")
import _pickle as pkl
with open(args.data_prefix + ".model_options.pkl", "rb") as fp:
	model_options = pkl.load(fp)
with open(args.data_prefix + ".cols.pkl", "rb") as fp:
	cols_dict = pkl.load(fp)

cat_dims = cols_dict["cat_dims"]
emb_dims = [(cat_dim, min(10, cat_dim // 2)) for cat_dim in cat_dims]
no_of_cont = len(cols_dict["cont_cols"])
lin_layer_sizes = [model_options['hidden1'], model_options['hidden2'], model_options['hidden3'], model_options['hidden4']]

#build model
import torch
import torch.cuda
from model import FeedForwardNN

print("building model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedForwardNN(emb_dims, no_of_cont=no_of_cont, lin_layer_sizes=lin_layer_sizes, output_size=model_options['output_dim']).to(device)

# loading best model from file
print("reloading model from file...")
model.load_state_dict(torch.load(args.data_prefix + ".best.model"))

model.test(isValidation=False)

# setup dataset
import pandas as pd
from model import TabularDataset
from torch.utils.data import DataLoader
print("importing dataset from pkl file...")
if args.isValidation:
	df_test = pd.read_pickle(args.data_prefix + ".val.pkl")
else:
	df_test = pd.read_pickle(args.data_prefix + ".test.pkl")
with open(args.data_prefix + ".cols.pkl", "rb") as fp:
	cols_dict = pkl.load(fp)

testset = TabularDataset(df_test, cat_cols=cols_dict["cat_cols"], output_col=cols_dict["output_cols"])
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=1)

# variables for score calculation
class_pred = [0.0 for i in range(2)]
class_correct = [0.0 for i in range(2)]
class_total = [0.0 for i in range(2)]

# evaluation
print("evaluation...")
for i, data in enumerate(testloader):
	y, cont_x, cat_x = data
	y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
	
	output = model(cont_x, cat_x)
	_, preds = torch.max(output.data, 1)

	c = (preds == y).squeeze()
	for j in range(preds.size(0)):
		class_pred[preds[j]] += 1

		label = y[j]
		class_correct[label] += c[j].item()
		class_total[label] += 1

	if i % 100 == 99:
		print("%d th records processed..." % ((i+1) * args.batch_size))

# score output
print("accuracy: %.4f"%(sum(class_correct)/sum(class_pred)))
print("precision: %.4f"%(class_correct[1]/class_pred[1]))
print("recall: %.4f"%(class_correct[1]/class_total[1]))
