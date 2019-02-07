from torch.utils.data import Dataset, DataLoader
'''
codes for defining subclasses of Dataset and DataLoader are adopted from Yashu Seth's blog:
https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
'''
class TabularDataset(Dataset):
	def __init__(self, data, cat_cols=None, output_col=None):
		self.n = data.shape[0]
		
		if cat_cols:
			self.cat_x = data[cat_cols].astype(np.int64).values
		else:
			self.cat_x = np.zeros((self.n, 1))

		if output_col: #only one output column
			self.y = data[output_col].astype(np.int64).values.reshape(-1,1)
		else:
			self.y = np.zeros((self.n, 1))

		self.cat_cols = cat_cols if cat_cols else []
		self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]

		if self.cont_cols:
			self.cont_x = data[self.cont_cols].astype(np.float32).values
		else:
			self.cont_x = np.zeros((self.n, 1))
	
	def __len__(self):
		return self.n
	
	def __getitem__(self, idx):
		return [self.y[idx], self.cont_x[idx], self.cat_x[idx]]

import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
	def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, output_size): #remove dropout parameters
		
		super().__init__()
		
		self.no_of_embs = sum([y for x,y in emb_dims])
		self.no_of_cont = no_of_cont

		#embedding layer
		self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
		
		#linear layers
		first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])
		
		self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i+1]) for i in range(len(lin_layer_sizes) - 1)])
		for lin_layer in self.lin_layers:
			nn.init.kaiming_normal_(lin_layer.weight)

		#output layer
		self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
		nn.init.kaiming_normal_(self.output_layer.weight.data)


	def forward(self, cont_data, cat_data):
		#cat_data through embedding layer
		if self.no_of_embs !=0:
			x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
			x = torch.cat(x, 1) #sequence of tensors given. () and [] both works!

		if self.no_of_cont != 0:
			
			if self.no_of_embs != 0:
				x = torch.cat([x, cont_data], 1) # concat with cont_data
			else:
				x = cont_data
		
		if self.no_of_embs == 0 and self.no_of_cont == 0 : 
			print("no_no_embs and no_of_cont are both zero")
			return None

		for lin_layer in self.lin_layers:			
			x = F.relu(lin_layer(x))

		return F.log_softmax(self.output_layer(x), dim=1)

def train(epoch):
	model.train()
	for eidx in range(epoch):
		running_loss = 0.0
		running_batches = 0
		for i, data in enumerate(trainloader):
			y, cont_x, cat_x = data
			y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
			#feedForward and loss
			preds = model(cont_x, cat_x)
			loss = criterion(preds, y)
		
			#backprop
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
			running_batches += 1
			
			if i % 100 == 99:
				print("[%d %d] loss: %.6f"%(eidx+1, i+1, running_loss / running_batches))
				running_loss = 0.0
				running_batches = 0
		validation(testloader)

def validation(dataloader):
	model.eval()
	'''
	codes for evaluating precision and recall is adopted from 
	https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
	'''
	class_pred = [0.0 for i in range(2)]
	class_correct = [0.0 for i in range(2)]
	class_total = [0.0 for i in range(2)]

	for i, data in enumerate(dataloader):
		y, cont_x, cat_x = data
		y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)	
		#preds = model(cont_x, cat_x) #save the result to preds
		#preds = (preds > 0.67).long() #change to _, preds = torch.max(output.data, 1)
		output = model(cont_x, cat_x)
		_, preds = torch.max(output.data, 1)

		c = (preds == y).squeeze()
		for j in range(preds.size(0)):
			class_pred[preds[j]] += 1

			label = y[j]
			class_correct[label] += c[j].item()
			class_total[label] += 1
		
		if i % 100 == 99:
			print("%d th records processed..."%((i+1)*batchsize))
			
	print("accuracy: %.3f"%(sum(class_correct)/sum(class_pred)))
	print("precision: %.3f"%(class_correct[1]/class_pred[1]))
	print("recall: %.3f"%(class_correct[1]/class_total[1]))
'''
codes fot importing dataset and preprocessing are also adopted from Yashu Seth's blog:
'''
#importing data from csv using pandas
import pandas as pd
import numpy as np
df_train = pd.read_csv("UNSW_NB15_training-set.csv") 
df_train = df_train.drop(["id", "attack_cat"], axis=1)

#import test data using pandas
df_test = pd.read_csv("UNSW_NB15_testing-set.csv")
df_test = df_test.drop(["id", "attack_cat"], axis=1)

#label encoding for categorical features
categorical_features = ["proto", "service", "state"]
output_feature = "label"
continuous_features = [col for col in df_train.columns if col not in categorical_features + [output_feature]]

vlist= ["ACC", "CLO"]
pos = np.flatnonzero(df_test["state"].isin(vlist))
df_test = df_test.drop(pos)

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for cat_col in categorical_features:
	label_encoders[cat_col] = LabelEncoder()
	df_train[cat_col] = label_encoders[cat_col].fit_transform(df_train[cat_col])
	df_test[cat_col] = label_encoders[cat_col].transform(df_test[cat_col])

#normalization of numeric data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df_train[continuous_features])
df_train[continuous_features] = scaler.transform(df_train[continuous_features])
df_test[continuous_features] = scaler.transform(df_test[continuous_features])

# shuffle the dataset
df_train = df_train.sample(frac=1).reset_index(drop=True) #adopted from stack_overflow - query: "shuffle dataframe rows"
#change name to df_train

# divide dataset into train and valid
trN = np.ceil(df_train.shape[0]*0.9).astype('int')
train_data = df_train[:trN] #df_train
valid_data = df_train[trN:] #df_train

#setting up dataset and dataLoader
batchsize = 1000

trainset = TabularDataset(train_data, cat_cols=categorical_features, output_col=output_feature)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=1)

validset = TabularDataset(valid_data, cat_cols=categorical_features, output_col=output_feature)
validloader = DataLoader(validset, batch_size=batchsize, shuffle=True, num_workers=1)

testset = TabularDataset(df_test, cat_cols=categorical_features, output_col=output_feature)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=True, num_workers=1)

#define embedding layer sizes
cat_dims = [df_train[col].nunique() for col in categorical_features]
emb_dims = [(cat_dim, min(50, cat_dim // 2)) for cat_dim in cat_dims]

#setup model
import torch.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedForwardNN(emb_dims, no_of_cont=39, lin_layer_sizes=[99, 64, 32, 16], output_size=2).to(device)

#setup variables for training the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
criterion = torch.nn.NLLLoss()

train(20)
validation(validloader)
validation(testloader)
