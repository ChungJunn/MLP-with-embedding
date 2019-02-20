'''
train.py
--------
set up model with given commandline arguments. Then train the model it goes through the following steps:
	1) take input arguments through argparse
	2) create dataloader with train and validation data
	3) build model
	4) train the model using model.train() function -> refer to model.py
	5) save results as files
'''
# argparser
import argparse

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)

# for loading input files
parser.add_argument("--data_prefix", type=str, default='unsw', help="str, prefix included in the names of training and validation dataset in pkl files")

# for model saving
parser.add_argument("--saveto", type=str, default='test', help="str, During training, the model writes its parameters into a file whenever the model's validation score imporves. 'saveto' is a string to include in the name of that file (in addition to prefix)")
parser.add_argument("--isReload", dest='isReload', action='store_true', help="no argument, when this option is specified the model will re-load existing model from files. Only possible when the model with the same 'data_prefix' and 'saveto' had been training previously. The model will simply adopt all training, model, and other parameters from saved files.")
parser.set_defaults(isReload=False)

# model architecture
parser.add_argument("--num_hidden", type=int, default=3, help="int, must be an integer within the range [1, 4]")
parser.add_argument("--hidden1", type=int, default=30, help="int")
parser.add_argument("--hidden2", type=int, default=20, help="int")
parser.add_argument("--hidden3", type=int, default=10, help="int")
parser.add_argument("--hidden4", type=int, default=None, help="int")
parser.add_argument("--output_dim", type=int, default=2, help="int")

# training parameters
parser.add_argument("--batch_size", type=int, default=128, help="int")
parser.add_argument("--loss", type=str, default='NLLLoss', help="str, name of loss function. Should match the names in torch.nn")
parser.add_argument("--optimizer", type=str, default='RMSprop', help="str, name of optimizer. Should match the names torch.optim")
parser.add_argument("--lr", type=float, default='0.002', help="float")
parser.add_argument("--max_epochs", type=int, default=5000, help="int")
parser.add_argument("--validFreq", type=int, default=1, help="int")
parser.add_argument("--patience", type=int, default=50, help="int")

args = parser.parse_args()

# in case isReload load model_options file
import _pickle as pkl
if args.isReload:
	print("re-loading model options...")
	with open(args.data_prefix + "." + args.saveto + ".model_options.pkl", "rb") as fp:
		model_options = pkl.load(fp)
	globals().update(model_options)
	isReload = args.isReload	
else:
	model_options = vars(args)
	globals().update(model_options)
	with open(data_prefix + "." + saveto + ".model_options.pkl", "wb") as fp: # write new model_options file
		pkl.dump(model_options, fp)

# load dataset and create dataloader from pkl file
print("importing datasets from pkl file...")
import pandas as pd
from model import TabularDataset
from torch.utils.data import DataLoader

df_train = pd.read_pickle(data_prefix + ".subtrain.pkl")
df_valid = pd.read_pickle(data_prefix + ".valid.pkl")

with open(data_prefix + ".cols.pkl", "rb") as fp:
	cols_dict = pkl.load(fp)

print("preparing dataloader")
trainset = TabularDataset(df_train, cat_cols=cols_dict["cat_cols"], output_col=cols_dict["output_cols"])
validset = TabularDataset(df_valid, cat_cols=cols_dict["cat_cols"], output_col=cols_dict["output_cols"])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=1)

# build model
print("building model...")
import torch
import torch.cuda
from model import FeedForwardNN

cat_dims = cols_dict["cat_dims"]
emb_dims = [(cat_dim, min(10, cat_dim // 2)) for cat_dim in cat_dims]
no_of_cont = len(cols_dict["cont_cols"])
lin_layer_sizes = [hidden1, hidden2, hidden3, hidden4]

#cat_dims = [df_train[col].nunique() for col in categorical_features]
#emb_dims = [(cat_dim, min(10, cat_dim // 2)) for cat_dim in cat_dims]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedForwardNN(emb_dims, no_of_cont=no_of_cont, lin_layer_sizes=lin_layer_sizes[:num_hidden], output_size=output_dim).to(device)

print(model_options)
print(model)

# case isReload, reload model parameters
if isReload:
	print("re-loading model parameters...")
	model.load_state_dict(torch.load(data_prefix + "." + saveto + ".model_best.pkl"))

# train for max_epochs
print("training...")
optimizer = "torch.optim." + optimizer
loss = "torch.nn." + loss
optimizer = eval(optimizer)(model.parameters(), lr=lr)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
criterion = eval(loss)()

model.train(data_prefix=data_prefix,
	saveto=saveto,
	isReload=isReload,
	trainloader=trainloader,
	validloader=validloader,
	device=device,
	max_epochs=max_epochs,
	criterion=criterion,
	optimizer=optimizer,
	validFreq=validFreq,
	patience=patience)

model.test(testloader=trainloader,
	data_prefix=data_prefix,
	loadfrom=saveto,
	device=device)
model.test(testloader=validloader,
	data_prefix=data_prefix,
	loadfrom=saveto,
	device=device)
