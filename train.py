'''
train.py
--------
set up model with given commandline arguments. Then train the model 

command line arguments using argparse
-------------------------------------
--data_prefix : str
name of dataset to train and evaluate on

--hidden1 : int
number of units in first hidden layer

--hidden2 : int
number of units in second hidden layer

--hidden3 : int
number of units in third hidden layer

--hidden4 : int
number of units in fourth hidden layer

--output_dim : int
number of units in output layer

--loss : str
name of loss function

--optimizer : str
name of optimizer

--lr : float
learning rate

--max_epochs : int
maximum number of epochs for training

--validFreq : int
number of epochs inbetween validation
'''
# argparser
import argparse

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--data_prefix", type=str, default='unsw', help="output dir")
parser.add_argument("--saveto", type=str, default='', help="output dir")
parser.add_argument("--num_hidden", type=int, default=4, help="output dir")
parser.add_argument("--hidden1", type=int, default=200, help="output dir")
parser.add_argument("--hidden2", type=int, default=100, help="output dir")
parser.add_argument("--hidden3", type=int, default=50, help="output dir")
parser.add_argument("--hidden4", type=int, default=20, help="output dir")
parser.add_argument("--output_dim", type=int, default=2, help="output dir")
parser.add_argument("--loss", type=str, default='NLLLoss', help="output dir")
parser.add_argument("--optimizer", type=str, default='Adadelta', help="output dir")
parser.add_argument("--lr", type=float, default='0.05', help="output dir")
parser.add_argument("--batch_size", type=int, default=128, help="output dir")
parser.add_argument("--max_epochs", type=int, default=1000, help="output dir")
parser.add_argument("--validFreq", type=int, default=5, help="output dir")
parser.add_argument("--patience", type=int, default=30, help="output dir")

args = parser.parse_args()

# load dataset and create dataloader from pkl file
print("importing datasets from pkl file...")
import pandas as pd
import _pickle as pkl
from model import TabularDataset
from torch.utils.data import DataLoader

df_train = pd.read_pickle(args.data_prefix + ".tr.pkl")
df_valid = pd.read_pickle(args.data_prefix + ".val.pkl")

with open(args.data_prefix + ".cols.pkl", "rb") as fp:
	cols_dict = pkl.load(fp)
if cols_dict["cat_cols"] == None:
	cols_dict["cat_cols"] = []

print("preparing dataloader")
trainset = TabularDataset(df_train, cat_cols=cols_dict["cat_cols"], output_col=cols_dict["output_cols"])
validset = TabularDataset(df_valid, cat_cols=cols_dict["cat_cols"], output_col=cols_dict["output_cols"])

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=1)

# build model
print("building model...")
import torch
import torch.cuda
from model import FeedForwardNN

cat_dims = cols_dict["cat_dims"]
emb_dims = [(cat_dim, min(10, cat_dim // 2)) for cat_dim in cat_dims]
no_of_cont = len(cols_dict["cont_cols"])
lin_layer_sizes = [args.hidden1, args.hidden2, args.hidden3, args.hidden4]

#cat_dims = [df_train[col].nunique() for col in categorical_features]
#emb_dims = [(cat_dim, min(10, cat_dim // 2)) for cat_dim in cat_dims]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedForwardNN(emb_dims, no_of_cont=no_of_cont, lin_layer_sizes=lin_layer_sizes[:args.num_hidden], output_size=args.output_dim).to(device)

# print model_option also as a file
model_options = vars(args)
print(model_options)
with open(args.data_prefix + "." + args.saveto + ".model_options.pkl", "wb") as fp:
	pkl.dump(vars(args), fp)

# train for max_epochs
print("training...")
optimizer = "torch.optim." + args.optimizer
loss = "torch.nn." + args.loss
optimizer = eval(optimizer)(model.parameters(), lr=args.lr)
criterion = eval(loss)()

model.train(data_prefix=args.data_prefix,
	saveto=args.saveto,
	trainloader=trainloader,
	validloader=validloader,
	device=device,
	max_epochs=args.max_epochs,
	criterion=criterion,
	optimizer=optimizer,
	validFreq=args.validFreq,
	patience=args.patience)

model.test(testloader=trainloader, device=device)
model.test(testloader=validloader, device=device)
