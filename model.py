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

import torch
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

	def train(self, data_prefix, saveto, isReload, trainloader, validloader, device, max_epochs, criterion, optimizer, validFreq, patience):
		
		# if isReload, load model_history file
		import _pickle as pkl	
		if isReload:
			print("re-loading model history...")
			with open(data_prefix + "." + saveto + ".model_history.pkl", "rb") as fp:
				model_history = pkl.load(fp)
			last_idx, last_err = model_history[-1]
			print("last training session:  eidx: %d err:%.4f" % (last_idx, last_err))
			start_idx = last_idx + 1
			best_err = last_err
		else:
			start_idx = 0
			best_err = 0
			model_history = []

		# initialize some variables and start training 
		val_err = 0.0
		bad_counter = 0
		estop = False

		for eidx in range(start_idx, max_epochs):
			for i, data in enumerate(trainloader):
				y, cont_x, cat_x = data
				y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
				preds = self.forward(cont_x, cat_x)
				loss = criterion(preds, y)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		# at each validFreq, validate model and save error
			if eidx % validFreq == 0:
				val_err = 0.0
				for i, data in enumerate(validloader):
					y, cont_x, cat_x = data
					y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
					preds = self.forward(cont_x, cat_x)
					loss = criterion(preds, y)

					val_err += loss.item()

				val_err /= i
				model_history.append((eidx, val_err))
				print("edix: %d, err: %.4f"%(eidx, val_err))

		# if achieving best error, save to history file and save model
				if eidx==0 or val_err <= best_err:
					print("above is the best model sofar...")
					best_err = val_err
					bad_counter = 0
					torch.save(self.state_dict(), data_prefix + "." + saveto + ".model_best.pkl")
					with open(data_prefix + "." + saveto + ".model_history.pkl", "wb") as fp:
						pkl.dump(model_history, fp) 

		# increment bad_counter and early-stop if appropriate
				if eidx > patience and val_err > best_err:
					bad_counter += 1
					if bad_counter > patience:
						estop = True
						break

			if estop:
				print("early stop!")
				break

	def test(self, testloader, data_prefix, loadfrom, device):
		'''
		codes adopted from https://pytorch.org tutorial 60 minutes blitz
		'''	
		# load best parameter from file
		self.load_state_dict(torch.load(data_prefix + "." + loadfrom + ".model_best.pkl"))

		# variables for score calculation
		class_pred = [0.0 for i in range(2)]
		class_correct = [0.0 for i in range(2)]
		class_total = [0.0 for i in range(2)]

		# evaluation
		print("evaluation...")
		for i, data in enumerate(testloader):
			y, cont_x, cat_x = data
			y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
			
			output = self.forward(cont_x, cat_x)
			_, preds = torch.max(output.data, 1)

			c = (preds == y).squeeze()
			for j in range(preds.size(0)):
				class_pred[preds[j]] += 1

				label = y[j]
				class_correct[label] += c[j].item()
				class_total[label] += 1

			if i % 100 == 99:
				print("%d th records processed..." % ((i+1) * testloader.batch_size))

		# score output
		print("accuracy: %.4f"%(sum(class_correct)/sum(class_pred)))
		print("precision: %.4f"%(class_correct[1]/class_pred[1]))
		print("recall: %.4f"%(class_correct[1]/class_total[1]))

