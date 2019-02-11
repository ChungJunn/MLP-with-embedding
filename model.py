from torch.utils.data import Dataset, DataLoader
import numpy as np
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
			self.y = data[output_col].astype(np.int64).values.squeeze()
		else:
			self.y = np.zeros((self.n, 1))

		self.cat_cols = cat_cols if cat_cols else []
		self.cont_cols = [col for col in data.columns if col not in self.cat_cols + output_col]

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
		
		#linear layer
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


	def learn(self,
		data_prefix="unsw",
		max_epochs=1000,
		batch_size=128,
		loss='NLLLoss',
		optimizer='SGD',
		lr=0.05,
		validFreq=5,
		patience=5):
	
		# load dataset and create dataloader from pkl file
		print("importing datasets from pkl file...")
		import pandas as pd
		import _pickle as pkl
		df_train = pd.read_pickle(data_prefix + ".tr.pkl")
		df_valid = pd.read_pickle(data_prefix + ".val.pkl")
		
		with open(data_prefix + ".cols.pkl", "rb") as fp:
			cols_dict = pkl.load(fp)

		trainset = TabularDataset(df_train, cat_cols=cols_dict["cat_cols"], output_col=cols_dict["output_cols"])
		validset = TabularDataset(df_valid, cat_cols=cols_dict["cat_cols"], output_col=cols_dict["output_cols"])

		trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
		validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=1)

		# create training history file
		print(str(self.lin_layers))
		
		# train for max_epochs
		print("training...")
		optimizer = "torch.optim." + optimizer
		loss = "torch.nn." + loss
		optimizer = eval(optimizer)(self.parameters(), lr=lr)
		criterion = eval(loss)()

		val_err = 0.0
		best_err = 0.0
		bad_counter = 0
		estop = False

		for eidx in range(max_epochs):
			for i, data in enumerate(trainloader):
				y, cont_x, cat_x = data
				y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
				preds = self.forward(cont_x, cat_x)
				loss = criterion(preds, y)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		# for each validFreq validate model and save error
			if eidx % validFreq == 0:
				val_err = 0.0
				for i, data in enumerate(validloader):
					y, cont_x, cat_x = data
					y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
					preds = self.forward(cont_x, cat_x)
					loss = criterion(preds, y)

					val_err += loss.item()
				
				val_err /= i
				print("edix: %d, err: %.4f"%(eidx, val_err))

		# if achieving best error, save to history file and save model
				if eidx==0 or val_err <= best_err:
					print("above is the best model sofar...")
					best_err = val_err
					torch.save(self.state_dict(), data_prefix + ".best.model")
		# increment bad_counter and early-stop if appropriate
				if eidx > patience and val_err > best_err:
					bad_counter += 1
					if bad_counter > patience:
						estop = True
						break
				
			if estop:
				print("early stop!")
				break

	def test(self, isReload=False, isValidation=False, data_prefix='unsw', batch_size=128):
		'''
		codes adopted from https://pytorch.org tutorial 60 minutes blitz
		'''
		# loading best model from file
		if isReload:
			print("reloading model from file...")
			self.load_state_dict(torch.load(data_prefix + ".best.model"))
		
		# setup dataset
		import pandas as pd
		import _pickle as pkl
		print("importing dataset from pkl file...")
		if isValidation:
			df_test = pd.read_pickle(data_prefix + ".val.pkl")
		else:
			df_test = pd.read_pickle(data_prefix + ".test.pkl")
		with open(data_prefix + ".cols.pkl", "rb") as fp:
			cols_dict = pkl.load(fp)

		testset = TabularDataset(df_test, cat_cols=cols_dict["cat_cols"], output_col=cols_dict["output_cols"])
		testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1)

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
				print("%d th records processed..." % ((i+1) * batch_size))

		# score output
		print("accuracy: %.4f"%(sum(class_correct)/sum(class_pred)))
		print("precision: %.4f"%(class_correct[1]/class_pred[1]))
		print("recall: %.4f"%(class_correct[1]/class_total[1]))

			
#define embedding layer sizes
cat_dims = [133, 13, 9]
emb_dims = [(cat_dim, min(10, cat_dim // 2)) for cat_dim in cat_dims]

#setup model
import torch.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedForwardNN(emb_dims, no_of_cont=39, lin_layer_sizes=[200, 100, 50, 20], output_size=2).to(device)
model.learn()
model.test(isValidation=True)
