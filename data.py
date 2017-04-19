import sys
import word2vec

def get_model_class(model):
	# class <model_name> from <model_name>.py
	module = __import__(model)
	model_class = getattr(module, model)
	return model_class

def sentence_length(sentence):
	return sentence.count(" ")+1

def index_and_align(sentence, length, generate_new_vector=False):
	words = sentence.split(" ")

	pad_size = length-len(words)

	if pad_size<0: # if the sentence is longer than the length we are trying to align it to
		words=words[0:length]
		pad_size=0

	# TODO instead of index 0, try to leave out words for which embedding does not exist
	word_indices = [word2vec.word_index(w, generate_new_vector) for w in words] + [0 for _ in range(pad_size)]
	return word_indices

def process_sentence(sentence):
	tokens = sentence.split(" ")
	unwanted = ".,;-\"'?!#$%&/()*+-=˝[]:><"
	words = [w.strip(unwanted) for w in tokens if w not in unwanted]

	# word2vec phrases joined by underscore
	word_to_phrases=[]
	i=0
	while i<len(words)-1:
		first=words[i]
		second=words[i+1]
		phrase = first+"_"+second
		if phrase in word2vec.indices_dictionary:
			word_to_phrases.append(phrase)
			i+=2
		else:
			word_to_phrases.append(first)
			i+=1


	sentence = " ".join(word_to_phrases)

	return sentence.encode(sys.stdout.encoding, errors="ignore").decode(sys.stdout.encoding).strip()

def get_dataset_classes(dataset_name):
	one_hot=None
	if dataset_name=="MR":
		one_hot = {
			"negative": [1, 0],
			"positive": [0, 1]
		}
	elif dataset_name=="TREC":
		one_hot = {
			"ABBR": [1, 0, 0, 0, 0, 0],
			"ENTY": [0, 1, 0, 0, 0, 0],
			"DESC": [0, 0, 1, 0, 0, 0],
			"HUM": [0, 0, 0, 1, 0, 0],
			"LOC": [0, 0, 0, 0, 1, 0],
			"NUM": [0, 0, 0, 0, 0, 1]
		}
	elif dataset_name=="pros-cons":
		one_hot = {
			"pros": [1, 0],
			"cons": [0, 1]
		}
	elif dataset_name=="20-newsgroup":
		one_hot = {
			"alt": [1, 0, 0, 0, 0, 0, 0],
			"comp": [0, 1, 0, 0, 0, 0, 0],
			"misc": [0, 0, 1, 0, 0, 0, 0],
			"rec": [0, 0, 0, 1, 0, 0, 0],
			"sci": [0, 0, 0, 0, 1, 0, 0],
			"soc": [0, 0, 0, 0, 0, 1, 0],
			"talk": [0, 0, 0, 0, 0, 0, 1]
		}
	elif dataset_name=="ReutersR8":
		one_hot = {
			"acq": [1, 0, 0, 0, 0, 0, 0, 0],
			"crude": [0, 1, 0, 0, 0, 0, 0, 0],
			"earn": [0, 0, 1, 0, 0, 0, 0, 0],
			"grain": [0, 0, 0, 1, 0, 0, 0, 0],
			"interest": [0, 0, 0, 0, 1, 0, 0, 0],
			"money-fx": [0, 0, 0, 0, 0, 1, 0, 0],
			"ship": [0, 0, 0, 0, 0, 0, 1, 0],
			"trade": [0, 0, 0, 0, 0, 0, 0, 1]
		}
	return one_hot

def load_dataset(dataset_name):
	base_path="./datasets/"

	train=[]
	test=[]
	max_sentence_length=0

	one_hot = get_dataset_classes(dataset_name)
	if one_hot==None:
		raise ValueError("Unknown dataset.")

	if dataset_name=="MR":
		with open(base_path+"rt-polaritydata/rt-polarity.neg", "r", encoding="utf8") as negative:
			i=0
			for sentence in negative:
				clean_sentence=process_sentence(sentence)
				if i<4000:
					length = sentence_length(clean_sentence)
					if length>max_sentence_length: max_sentence_length=length

					train.append((clean_sentence, one_hot["negative"]))
				else:
					test.append((clean_sentence, one_hot["negative"]))

				i+=1


		with open(base_path+"rt-polaritydata/rt-polarity.pos", "r", encoding="utf8") as positive:
			i=0
			for sentence in positive:
				clean_sentence=process_sentence(sentence)
				if i<4000:
					length = sentence_length(clean_sentence)
					if length>max_sentence_length: max_sentence_length=length

					train.append((clean_sentence, one_hot["positive"]))
				else:
					test.append((clean_sentence, one_hot["positive"]))

				i+=1

		return train, test, 2, one_hot, max_sentence_length

	elif dataset_name=="TREC":

		with open(base_path+"TREC/train_5500.label", "r", encoding="utf8") as train_set:
			for line in train_set:
				tokens = line.split(" ")
				category = tokens[0].split(":")[0]
				sentence = " ".join(tokens[1:])

				clean_sentence = process_sentence(sentence)
				length = sentence_length(clean_sentence)
				if length>max_sentence_length: max_sentence_length=length

				train.append((clean_sentence, one_hot[category]))

		with open(base_path+"TREC/TREC_10.label", "r", encoding="utf8") as test_set:
			for line in test_set:
				tokens = line.split(" ")
				category = tokens[0].split(":")[0]
				sentence = " ".join(tokens[1:])

				clean_sentence = process_sentence(sentence)

				test.append((clean_sentence, one_hot[category]))

		return train, test, 6, one_hot, max_sentence_length

	elif dataset_name=="pros-cons":

		with open(base_path+"pros-cons/IntegratedPros.txt", "r", encoding="utf8") as pros:
			i=0
			for sentence in pros:
				clean_sentence = sentence.strip(" ").replace("<Pros>", "").replace("</Pros>", "")
				if i<17200:
					length = sentence_length(clean_sentence)
					if length>max_sentence_length: max_sentence_length=length

					train.append((clean_sentence, one_hot["pros"]))
				else:
					test.append((clean_sentence, one_hot["pros"]))

				i+=1

		with open(base_path+"pros-cons/IntegratedCons.txt", "r", encoding="utf8") as cons:
			i=0
			for sentence in cons:
				clean_sentence = sentence.strip(" ").replace("<Cons>", "").replace("</Cons>", "")
				if i<17200:
					length = sentence_length(clean_sentence)
					if length>max_sentence_length: max_sentence_length=length

					train.append((clean_sentence, one_hot["cons"]))
				else:
					test.append((clean_sentence, one_hot["cons"]))

				i+=1

		return train, test, 2, one_hot, max_sentence_length

	elif dataset_name=="20-newsgroup":

		with open(base_path+"20-newsgroup/20ng-train-no-stop.txt", "r", encoding="utf8") as train_set:
			for line in train_set:
				tokens = line.split(" ")
				category = tokens[0].split(".")[0]
				sentence = " ".join(tokens[1:])

				clean_sentence = process_sentence(sentence)
				length = sentence_length(clean_sentence)
				if length>max_sentence_length: max_sentence_length=length

				train.append((clean_sentence, one_hot[category]))

		with open(base_path+"20-newsgroup/20ng-test-no-stop.txt", "r", encoding="utf8") as test_set:
			for line in test_set:
				tokens = line.split(" ")
				category = tokens[0].split(".")[0]
				sentence = " ".join(tokens[1:])

				test.append((process_sentence(sentence), one_hot[category]))

		return train, test, 7, one_hot, max_sentence_length

	elif dataset_name=="ReutersR8":

		with open(base_path+"Reuters-21578_R8/r8-train-no-stop.txt", "r", encoding="utf8") as train_set:
			for line in train_set:
				tokens = line.split("\t")
				category = tokens[0]
				sentence = tokens[1]

				clean_sentence = process_sentence(sentence)
				length = sentence_length(clean_sentence)
				if length>max_sentence_length: max_sentence_length=length

				train.append((clean_sentence, one_hot[category]))

		with open(base_path+"Reuters-21578_R8/r8-test-no-stop.txt", "r", encoding="utf8") as test_set:
			for line in test_set:
				tokens = line.split("\t")
				category = tokens[0]
				sentence = tokens[1]

				test.append((process_sentence(sentence), one_hot[category]))

		return train, test, 8, one_hot, max_sentence_length

def save_model(model, step=None):
	from os import makedirs

	base_path = "./models/"+model.model_name+"/"

	try:
		makedirs(base_path)
	except FileExistsError:
		pass

	model.saver.save(sess=model.session, save_path=base_path+model.model_name, global_step=step)

def load_model(session, model, model_name):
	import tensorflow as tf

	model_class = get_model_class(model)

	neural_network = model_class()
	neural_network.session=session
	neural_network.model_name=model_name

	base_path = "./models/"+model_name+"/"

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

		partitions.append((start,end))
		if stop:
			break
		i+=partition_size
	return partitions
