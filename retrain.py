'''
retrain.py
--------
retrain the model on the whole train dataset given the best_epoch
'''
# argparser
import argparse

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)

# for loading input files
parser.add_argument("--data_prefix", type=str, default='unsw', help="str, prefix included in the names of training and validation dataset in pkl files")

# for model saving
parser.add_argument("--saveto", type=str, default='test', help="str, During training, the model writes its parameters into a file whenever the model's validation score imporves. 'saveto' is a string to include in the name of that file (in addition to prefix)")

parser.add_argument("--max_epochs", type=int, default='', help="int, how many epochs to train the model")

args = parser.parse_args()

# in case isReload load model_options file
import _pickle as pkl
print("re-loading model options...")
with open(args.data_prefix + "." + args.saveto + ".model_options.pkl", "rb") as fp:
	model_options = pkl.load(fp)
globals().update(model_options)
max_epochs = args.max_epochs

# load dataset and create dataloader from pkl file
print("importing datasets from pkl file...")
import pandas as pd
from model import TabularDataset
from torch.utils.data import DataLoader

df_train = pd.read_pickle(data_prefix + ".train.pkl")

with open(data_prefix + ".cols.pkl", "rb") as fp:
	cols_dict = pkl.load(fp)

print("preparing dataloader")
trainset = TabularDataset(df_train, cat_cols=cols_dict["cat_cols"], output_col=cols_dict["output_cols"])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

# build model
print("building model...")
import torch
import torch.cuda
from model import FeedForwardNN

cat_dims = cols_dict["cat_dims"]
emb_dims = [(cat_dim, min(10, cat_dim // 2)) for cat_dim in cat_dims]
no_of_cont = len(cols_dict["cont_cols"])
lin_layer_sizes = [hidden1, hidden2, hidden3, hidden4]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedForwardNN(emb_dims, no_of_cont=no_of_cont, lin_layer_sizes=lin_layer_sizes[:num_hidden], output_size=output_dim).to(device)

print(model_options)
print(model)

# train for max_epochs
print("training...")
optimizer = "torch.optim." + optimizer
loss = "torch.nn." + loss
optimizer = eval(optimizer)(model.parameters(), lr=lr)
criterion = eval(loss)()

running_loss = 0.0
running_samples = 0

for eidx in range(max_epochs):
	for i, data in enumerate(trainloader):
		y, cont_x, cat_x = data
		y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
		preds = model.forward(cont_x, cat_x)
		loss = criterion(preds, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

		if i % 1000 == 999:
			print("[%d th epoch %d th data] loss : %.4f" % (eidx+1, i+1, running_loss / i))
			running_loss = 0.0

torch.save(model.state_dict(), data_prefix + "." + saveto + ".model_best.pkl")
