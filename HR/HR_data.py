import sys
import os
import HR_vocabulary

def get_model_class(model):
	# class <model_name> from <model_name>.py
	module = __import__(model)
	model_class = getattr(module, model)
	return model_class

def sentence_length(sentence):
	return sentence.count(" ")+1

def index_and_align(sentence, length):
	words = sentence.split(" ")

	pad_size = length-len(words)

	if pad_size<0: # if the sentence is longer than the length we are trying to align it to
		words=words[0:length]
		pad_size=0

	# TODO instead of index 0, try to leave out words for which embedding does not exist
	word_indices = [HR_vocabulary.word_index(w) for w in words] + [0 for _ in range(pad_size)]
	return word_indices

def process_sentence(sentence, max_size):
	text = sentence.lower()
	text = text.replace("\n", " ")
	text = text.replace("\t", " ")
	text = text.replace(u"\u00A0", " ")
	text = "".join([c for c in text if (c in " " or c.isalnum()) and not c.isdigit()])
	word_tokens = text.split(" ")
	text = " ".join([token for token in word_tokens if token.strip()!=""])
	word_tokens = text.split(" ")
	if max_size!=0 and len(word_tokens)>max_size:
		word_tokens = word_tokens[0:max_size]
	text = " ".join(word_tokens)
	return text

def get_dataset_classes(dataset_name):
	one_hot=None
	if dataset_name=="vecernji":
		one_hot = {
			"auti":			[1, 0, 0, 0, 0, 0, 0, 0, 0],
			"biznis":		[0, 1, 0, 0, 0, 0, 0, 0, 0],
			"kultura":		[0, 0, 1, 0, 0, 0, 0, 0, 0],
			"lifestyle":	[0, 0, 0, 1, 0, 0, 0, 0, 0],
			"showbiz":		[0, 0, 0, 0, 1, 0, 0, 0, 0],
			"sport":		[0, 0, 0, 0, 0, 1, 0, 0, 0],
			"techsci":		[0, 0, 0, 0, 0, 0, 1, 0, 0],
			"vijesti":		[0, 0, 0, 0, 0, 0, 0, 1, 0],
			"zagreb":		[0, 0, 0, 0, 0, 0, 0, 0, 1]
		}
	elif dataset_name=="24-sata":
		one_hot = {
			"lifestyle":			[1, 0, 0, 0, 0],
			"news":					[0, 1, 0, 0, 0],
			"show":					[0, 0, 1, 0, 0],
			"sport":				[0, 0, 0, 1, 0],
			"tech":					[0, 0, 0, 0, 1]
		}
	return one_hot

vocabulary=dict()

def load_dataset(dataset_name, vocabulary_size, max_document_size):
	base_path="./HR_datasets/"

	train=[]
	test=[]
	max_sentence_length=0

	one_hot = get_dataset_classes(dataset_name)
	if one_hot==None:
		raise ValueError("Unknown dataset.")

	category_dirs = [os.path.join(base_path, dataset_name, d) for d in os.listdir(os.path.join(base_path, dataset_name)) if os.path.isdir(os.path.join(base_path, dataset_name, d))]

	for category_dir in category_dirs:
		category_name = os.path.basename(category_dir)
		train_dir = os.path.join(category_dir, "train")
		test_dir = os.path.join(category_dir, "test")

		train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
		test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]

		for train_file in train_files:
			with open(train_file, "r", encoding="utf8") as f:
				content = f.read().strip("\n")
				clean_sentence = process_sentence(content, max_document_size)
				for word in clean_sentence.split(" "):
					if word not in vocabulary:
						vocabulary[word]=1
					else:
						vocabulary[word]+=1
				length = sentence_length(clean_sentence)
				if length>max_sentence_length: max_sentence_length=length
				train.append((clean_sentence, one_hot[category_name]))

		for test_file in test_files:
			with open(test_file, "r", encoding="utf8") as f:
				clean_sentence = process_sentence(f.read().strip("\n"), max_document_size)
				test.append((clean_sentence, one_hot[category_name]))

	if vocabulary_size!=0:
		mfw = sorted(vocabulary, key=vocabulary.get, reverse=True)[0:vocabulary_size]
	else:
		mfw = sorted(vocabulary, key=vocabulary.get, reverse=True)

	mfw.insert(0, "__NULA__")

	HR_vocabulary.load_indices(mfw)

	return train, test, len(one_hot.keys()), one_hot, max_sentence_length

def save_model(model, step=None):
	from os import makedirs

	base_path = "./models/"+model.model_name+"/"

	try:
		makedirs(base_path)
	except FileExistsError:
		pass

	# save all indices in word2vec.indices_dictionary.values() greater than word2vec.vocabulary_size
	with open(base_path+"indices", "w", encoding="utf8") as indices:
		for word in HR_vocabulary.indices_dictionary:
			index = HR_vocabulary.indices_dictionary[word]

			print(word+"\t"+str(index), file=indices)


	model.saver.save(sess=model.session, save_path=base_path+model.model_name, global_step=step)

def load_model(session, model, model_name):
	import tensorflow as tf

	model_class = get_model_class(model)

	neural_network = model_class()
	neural_network.session=session
	neural_network.model_name=model_name

	base_path = "./models/"+model_name+"/"

	with open(base_path+"indices", "r", encoding="utf-8") as indices:
		for line in indices:
			tokens = line.strip("\n").split("\t")
			word = tokens[0]
			index = tokens[1]
			HR_vocabulary.indices_dictionary[word]=index

	latest = tf.train.latest_checkpoint(base_path)

	neural_network.saver = tf.train.import_meta_graph(latest+".meta")
	neural_network.saver.restore(neural_network.session, latest)

	ops = [n.name for n in neural_network.session.graph.as_graph_def().node if '/' not in n.name]

	for op in ops:
		op_outputs = neural_network.session.graph.get_operation_by_name(op).outputs
		if op_outputs==None or len(op_outputs)<1: continue
		setattr(neural_network, op, neural_network.session.graph.get_operation_by_name(op).outputs[0])

	return neural_network

def generate_partitions(total_size, partition_size):
	partitions=[]
	i=0
	stop=False
	while True:
		start=i
		end=start+partition_size
		if end>total_size:
			end=total_size
			stop=True

		if start!=end:
			partitions.append((start,end))

		if stop:
			break
		i+=partition_size
	return partitions
