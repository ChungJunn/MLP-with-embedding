'''
preprocess.py
-------------
load  train dataset and test datasets csv files into pandas dataframe and preprocess them in the following ways:

1) label encoding of categorical data - preparing for embedding
2) normalize numeric data (scaler is fit to training set)
3) divide train dataset into train and valid dataset with given ratio

the results are preprocessed data in dataframes. They are saved into three separate pkl files using pickle
'''

# argparser
import os
import argparse
parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)

# CSV files
parser.add_argument("--train_data", type=str, default='UNSW_NB15_training-set.csv', help="str, path to train csv file")
parser.add_argument("--test_data", type=str, default='UNSW_NB15_testing-set_unknowns_removed.csv', help="str, path to test csv file")

# column(feature) information
parser.add_argument("--cont_cols", type=str, default='cont_cols.csv', help="str, path to a file which specify the column names of output features")
parser.add_argument("--cat_cols", type=str, default='cat_cols.csv', help="str, path to a file which specify the column names of categorical features")
parser.add_argument("--output_cols", type=str, default='output_cols.csv', help="str, path to a file which specify the column names of output features")

# other parameters
parser.add_argument("--data_prefix", type=str, default='unsw', help="str, prefix to be included in names of pkl files")
parser.add_argument("--normalization", type=str, default='standard', help="str ('standard' or 'minmax'), spcify the type of normalization scheme")
parser.add_argument("--val_ratio", type=float, default=0.1, help="float, specify how much portion of training dataset will be separated for validatin dataset")

args = parser.parse_args()
print(args)


# read in column names from csv files
print("reading column information from files...")
with open(args.cont_cols, "rt") as fp:
	cont_cols = [col.strip() for col in fp.readline().split(',')]
with open(args.cat_cols, "rt") as fp:
	cat_cols = [col.strip() for col in fp.readline().split(',')]
with open(args.output_cols, "rt") as fp:
	output_cols = [col.strip() for col in fp.readline().split(',')]

if cat_cols[0]=='none':
	cat_cols = []

cols_dict = {"cont_cols":cont_cols, "cat_cols":cat_cols, "output_cols":output_cols}
used_cols = cont_cols + cat_cols + output_cols

# import data from csv file using pandas
print("importing datasets...")
import pandas as pd
df_train = pd.read_csv(args.train_data)
df_train = df_train.drop([col for col in df_train.columns if col not in used_cols], axis=1)
df_test = pd.read_csv(args.test_data)
df_test = df_test.drop([col for col in df_test.columns if col not in used_cols], axis=1)

# save number of unique values for each cat_cols for train.py
cols_dict["cat_dims"] = [df_train[col].nunique() for col in cat_cols]

# label encoding for categorical features
print("label encoding...")
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for cat_col in cat_cols:
	label_encoders[cat_col] = LabelEncoder()
	df_train[cat_col] = label_encoders[cat_col].fit_transform(df_train[cat_col])
	df_test[cat_col] = label_encoders[cat_col].transform(df_test[cat_col])


# normalize continuous data
import sys
print("normalizing datasets...")
if args.normalization not in ['standard', 'minmax']: # handling invalid input
	sys.exit("--normalization arguement should be either 'standard' or 'minmax'")
elif args.normalization == 'standard':
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
else:
	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler()
scaler.fit(df_train[cont_cols])
df_train[cont_cols] = scaler.transform(df_train[cont_cols])
df_test[cont_cols] = scaler.transform(df_test[cont_cols])

# divide dataset into train and valid
print("dividing training dataset into train and valid")
import numpy as np
df_train = df_train.sample(frac=1).reset_index(drop=True) #shuffle

valN = np.ceil(df_train.shape[0] * args.val_ratio).astype('int')

valid_data = df_train[:valN]
train_data = df_train[valN:]

# print shapes of resulting dataframes
print("showing shapes of resulting dataframes")
print("\ttrain : ", train_data.shape)
print("\tvalid : ", valid_data.shape)
print("\ttest : ", df_test.shape)

#save results into pkl file
print("saving into pkl files...")
import _pickle as pkl
train_data.to_pickle(args.data_prefix + ".tr.pkl")
valid_data.to_pickle(args.data_prefix + ".val.pkl")
df_test.to_pickle(args.data_prefix + ".test.pkl")

with open(args.data_prefix + ".cols.pkl", "wb") as fp:
	pkl.dump(cols_dict, fp)

print("done")
