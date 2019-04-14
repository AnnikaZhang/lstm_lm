import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from scipy import stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("current device", device)
log = open("log.txt", "a+")
parser = argparse.ArgumentParser()
parser.add_argument('-b', action = "store_true", default = False)
parser.add_argument('-h', action = "store_true", default = False)
parser.add_argument('-r', action = "store", type = int)
parser.add_argument('-epoch', action = "store", type = int)
parser.add_argument('-f', action = "store", type = float)
args = parser.parse_args()
r = args.r
f = args.f
BINARY_LOSS_FLAG  = args.b
HINGE_LOSS_FLAG = args.h
total_epoch = args.epoch
log.write("Binary Loss enabled? %i \n" % BINARY_LOSS_FLAG)
log.write("number of epochs %i \n" % total_epoch)
log.write("number of sample %i \n"% r)
log.write("distribution power %f \n"% f)
model_name = "models/"
if HINGE_LOSS_FLAG:
		model_name += "best_hinge_loss_r"+str(r) + "_f" + str(f)
		print("Entering hinge loss mode")
elif BINARY_LOSS_FLAG:
	model_name += "best_binary_loss_r"+str(r) + "_f" + str(f)
	print("Entring binary loss mode")
else:
	model_name += "best_log_loss"

class LSTM_Net(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim):
		super(LSTM_Net, self).__init__()
		self.lstm = nn.LSTM(in_dim, hidden_dim)
		self.l = nn.Linear(hidden_dim, out_dim)
	def forward(self, x):
		lstm_out, _ = self.lstm(x)
		## only return the prediction of the last 
		return self.l(lstm_out.view(x.size()[0], -1))[0:-1]

class best_model():
	def __init__(self):
		self.dev = 0
		self.test = 0
		self.n_sentence = 0
		self.time = 0


def load_vocab():
## return a hash table for word look up
	voc = open("31210-s19-hw1/bobsue.voc.txt")
	table = {}
	wordlist=[]
	count = 0
	for line in voc:
		word = line.strip("\n")
		wordlist.append(word)
		table[word] = count
		count+=1
	return wordlist, table

def one_hot(word, table):
	vec = torch.zeros(len(table)).cuda()
	vec[table[word]] = 1
	return vec

def sentence_input(sentence, table):
	ls = [one_hot(w, table) for w in sentence]
	x = torch.cat(ls, dim = 0)
	return x.view(len(sentence), 1, -1)

def get_target(sentence, table):
	ls = [table[w] for w in sentence[1:]]
	return torch.LongTensor(ls).cuda()


def count_test(file, table):
	data = open("31210-s19-hw1/" + file)
	count = 0
	for sentence in data:
		words = sentence.strip("\n").split()
		for word in words:
			if word not in table:
				count += 1
	return count

def load_sentence(file):
	data = open("31210-s19-hw1/" + file)
	res = []
	for sentence in data:
		words = sentence.strip("\n").split()
		res.append(words)
	return res

def load_large_sentence(file):
	data = open("31210-s19-hw1/" + file)
	collection =[]
	target = []
	for sentence in data:
		words = sentence.strip("\n").split()
		start = words[1:].index("<s>") + 1
		label = words[start + 1:]
		collection.append(words)
		target.append(label)
	return collection, label

def test(data, t, verbose):
	count = 0
	correct = 0
	for index, s in enumerate(data):
		x = sentence_input(s, t)
		target = get_target(s, t)
		count += len(target)
		with torch.no_grad():
			y = net(x)
			result = torch.argmax(y, dim = 1)
			
			if verbose == True and index == 50:
				print("expected:", " ".join([wl[word] for word in target]))
				print("get:     ", " ".join([wl[word] for word in result]))

			for i in range(len(result)):
				if result[i] == target[i]:
					correct += 1
	return correct/count

def dev_test(best, dev_data, test_data, t, n_sentence, verbose):
	dev_result = test(dev_data, t, verbose)
	test_result = test(test_data, t, verbose)
	if dev_result > best.dev:
		print("dev result", dev_result)
		print("test result", test_result)
		best.dev = dev_result
		best.test = test_result
		best.time = time.time()
		best.n_sentence = n_sentence
		torch.save(net, model_name)
	return

def unigram(data,f):
		vector = np.zeros(vocab_size)
		for sentence in data:
				for word in sentence[1:]:
						index = t[word]
						vector[index]+=1
		vector = vector / np.sum(vector)
		vector = np.power(vector, f)
		vector = vector / np.sum(vector)
		xk = np.arange(vocab_size)
		distribution= stats.rv_discrete(name = 'custom', values = (xk, vector))
		return distribution



def one_hot_batch(sample, num_label):
	## sampke is a long tensor
	seq_len = len(sample)
	vector = torch.FloatTensor(seq_len, num_label).cuda()
	vector.zero_()
	vector.scatter_(1, sample,1)
	return vector

def binary_loss(output, target, r, num_label):
	## output dim: num_words * vocab_size
	## create one-hot vector for negative sampling
		if BINARY_LOSS_FLAG:
				R = unigram_dist.rvs(size = r)
				#print(R[0:100])
				sample = torch.LongTensor(R).cuda()
				sample = sample.unsqueeze(1)
		else:
				sample = torch.LongTensor(r,1).random_(1, num_label).cuda()
		one_hot_sample = one_hot_batch(sample, num_label)
		neg = F.sigmoid(torch.mul(output, torch.sum(one_hot_sample, dim = 0)))
		neg = torch.sum(torch.log(1-neg), dim = 1)
		## create one-hot vector for target
		one_hot_target = one_hot_batch(target.unsqueeze(1), num_label)
		loss = - torch.log(F.sigmoid(torch.sum(torch.mul(output, one_hot_target), dim = 1))) - neg
	#print(loss.requires_grad)
		return loss

def hinge_loss(output, target, num_label):
		R = unigram_dist.rvs(size = r)
		sample = torch.LongTensor(R).cuda()
		sample = sample.unsqueeze(1)
		one_hot_sample = one_hot_batch(sample, num_label)
		one_hot_sample = one_hot_sample.permute(1,0)
		neg_score = torch.mm(output, one_hot_sample) 
		one_hot_target = one_hot_batch(target.unsqueeze(1), num_label)
		target_score = torch.sum(torch.mul(output, one_hot_target), dim = 1).repeat(r,1).permute(1,0)
		loss = torch.sum(F.relu(1 - target_score + neg_score), dim = 1)
		return loss


#BINARY_LOSS_FLAG = False
wl, t = load_vocab()
vocab_size = len(wl)
## Feed a sequence of one-hot vectors to the RNN network
## in_dim is the vocab size since we feed one-hot vector
## hidden_dim is 200
## out_dim is the vocab size
net = LSTM_Net(vocab_size, 200, vocab_size).cuda()
optimizer = optim.Adam(net.parameters())
train_data= load_sentence("bobsue.lm.train.txt")
dev_data= load_sentence("bobsue.lm.dev.txt")
test_data= load_sentence("bobsue.lm.test.txt")
criterion = nn.CrossEntropyLoss(reduce = False)
unigram_dist = unigram(train_data,f)
print("number of training data", len(train_data))
best = best_model()
time_per_sentence = 0
start_train_time = time.time()
n_sentence = 0
for n_epoch in range(total_epoch):
	print("number of epoch is", n_epoch)
	acc_loss = 0
	count = 0
	processed_sentence = 0
	start_epoch = time.time()
	for (i, sentence) in enumerate(train_data):
		n_sentence+=1
		if i % 2000 == 1:
			dev_test(best, dev_data, test_data, t, n_sentence, False)
		optimizer.zero_grad()
		net.zero_grad()
		x = sentence_input(sentence, t)
		y = net(x)
		target = get_target(sentence, t)
		if HINGE_LOSS_FLAG:
			loss = hinge_loss(y, target, r, vocab_size)
		elif BINARY_LOSS_FLAG:
			loss = binary_loss(y, target, r, vocab_size)
		else:
			loss = criterion(y, target)
		count += len(loss)
		acc_loss += torch.sum(loss)
		loss.sum().backward()
		optimizer.step()
	end_epoch = time.time()
	temp = (end_epoch - start_epoch)/len(train_data)
	print("process time per sentence", temp)
	time_per_sentence += (end_epoch - start_epoch)/len(train_data)
	print("log loss", acc_loss/count)
	dev_test(best, dev_data, test_data, t, n_sentence, False)
	print("best dev with test", best.dev, best.test)
time_per_sentence /= total_epoch
log.write("best dev is %f, best test is %f \n" % (best.dev, best.test))
log.write("average process time for a sentence: %f \n" % time_per_sentence)
log.write("number of sentences to reach best dev: %i \n"% best.n_sentence)
temp = (best.time - start_train_time)/60
log.write("minites to read best dev: %f \n" % temp)
log.write("\n \n")


	


