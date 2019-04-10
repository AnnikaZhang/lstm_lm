import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTM_Net(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim):
		super(LSTM_Net, self).__init__()
		self.lstm = nn.LSTM(in_dim, hidden_dim)
		self.l = nn.Linear(hidden_dim, out_dim)
	def forward(self, x):
		lstm_out, _ = self.lstm(x)
		## only return the prediction of the last 
		return self.l(lstm_out.view(x.size()[0], -1))[0:-1]
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
	vec = torch.zeros(len(table))
	vec[table[word]] = 1
	return vec

def sentence_input(sentence, table):
	ls = [one_hot(w, table) for w in sentence]
	x = torch.cat(ls, dim = 0)
	return x.view(len(sentence), 1, -1)

def get_target(sentence, table):
	ls = [table[w] for w in sentence[1:]]
	return torch.LongTensor(ls)


def count_test(file, table):
	data = open("31210-s19-hw1/" + file)
	count = 0
	for sentence in data:
		words = sentence.strip("\n").split(" ")
		for word in words:
			if word not in table:
				count += 1
	return count

def load_sentence(file):
	data = open("31210-s19-hw1/" + file)
	res = []
	for sentence in data:
		words = sentence.strip("\n").split(" ")
		res.append(words)
	return res

def test(data, t):
	count = 0
	correct = 0
	for index, s in enumerate(data):
		x = sentence_input(s, t)
		target = get_target(s, t)
		count += len(target)
		with torch.no_grad():
			y = net(x)
			result = torch.argmax(y, dim = 1)
			
			if index %500 == 10:
				print("expected:", " ".join([wl[word] for word in target]))
				print("get:     ", " ".join([wl[word] for word in result]))

			for i in range(len(result)):
				if result[i] == target[i]:
					correct += 1
	return correct/count

def dev_test(best, dev_data, test_data, t):
	dev_result = test(dev_data, t)
	test_result = test(test_data, t)
	if dev_result > best[0]:
		print("dev result", dev_result)
		print("test result", test_result)
		best[0] = dev_result
		best[1] = test_result
	return

def error_count(test_data, net, wl):
	error_table = {}
	## count the ocuring word pair (golden standard, predicted result)
	for s in test_data:
		x = sentence_input(s, t)
		target = get_target(s, t)
		with torch.no_grad():
			y = net(x)
			result = torch.argmax(y, dim = 1)
			for i in range(len(result)):
				if result[i]!=target[i]:
					pair = (wl[target[i]], wl[result[i]])
					if pair in error_table:
						error_table[pair] +=1
					else:
						error_table[pair]= 1
	ls = []
	for pair in error_table:
		triple = (error_table[pair], pair[0], pair[1])
		ls.append(triple)
	print(ls[0:10])
	new_ls=sorted(ls, reverse = True)
	print(new_ls[0:10])
	for i in range(35):
		triple = new_ls[i]
		print("occured", triple[0], "times:", triple[1], triple[2])
	return



net = torch.load("models/best_log_loss", map_location = 'cpu')


wl, t = load_vocab()
vocab_size = len(wl)
## Feed a sequence of one-hot vectors to the RNN network
## in_dim is the vocab size since we feed one-hot vector
## hidden_dim is 200
## out_dim is the vocab size
#optimizer = optim.SGD(net.parameters(), lr=0.01)
train_data= load_sentence("bobsue.lm.train.txt")
dev_data= load_sentence("bobsue.lm.dev.txt")
test_data= load_sentence("bobsue.lm.test.txt")
error_count(test_data, net, wl)
#best = [0,0]
#dev_test(best, dev_data, test_data, t)