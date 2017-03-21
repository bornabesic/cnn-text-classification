import sys
import word2vec

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

	return sentence.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding).strip()

def load(dataset_name):
	base_path="./datasets/"

	train=[]
	test=[]

	if dataset_name=="MR":
		with open(base_path+"rt-polaritydata/rt-polarity.neg") as negative:
			i=0
			for sentence in negative:
				if i<4000:
					train.append((process_sentence(sentence), [1, 0]))
				else:
					test.append((process_sentence(sentence), [1, 0]))

				i+=1


		with open(base_path+"rt-polaritydata/rt-polarity.pos") as positive:
			i=0
			for sentence in positive:
				if i<4000:
					train.append((process_sentence(sentence), [0, 1]))
				else:
					test.append((process_sentence(sentence), [0, 1]))

				i+=1

		return train, test, 2

	elif dataset_name=="TREC":
		one_hot = {
			"ABBR": [1, 0, 0, 0, 0, 0],
			"ENTY": [0, 1, 0, 0, 0, 0],
			"DESC": [0, 0, 1, 0, 0, 0],
			"HUM": [0, 0, 0, 1, 0, 0],
			"LOC": [0, 0, 0, 0, 1, 0],
			"NUM": [0, 0, 0, 0, 0, 1]
		}

		with open(base_path+"TREC/train_5500.label") as train_set:
			for line in train_set:
				tokens = line.split(" ")
				category = tokens[0].split(":")[0]
				sentence = " ".join(tokens[1:])

				train.append((process_sentence(sentence), one_hot[category]))

		with open(base_path+"TREC/TREC_10.label") as test_set:
			for line in test_set:
				tokens = line.split(" ")
				category = tokens[0].split(":")[0]
				sentence = " ".join(tokens[1:])

				test.append((process_sentence(sentence), one_hot[category]))

		return train, test, 6

	elif dataset_name=="pros-cons":
		with open(base_path+"pros-cons/IntegratedPros.txt") as pros:
			i=0
			for sentence in pros:
				clean_sentence = sentence.strip(" ").replace("<Pros>", "").replace("</Pros>", "")
				if i<17200:
					train.append((process_sentence(clean_sentence), [1, 0]))
				else:
					test.append((process_sentence(clean_sentence), [1, 0]))

				i+=1
		
		with open(base_path+"pros-cons/IntegratedCons.txt") as cons:
			i=0
			for sentence in cons:
				clean_sentence = sentence.strip(" ").replace("<Cons>", "").replace("</Cons>", "")
				if i<17200:
					train.append((process_sentence(clean_sentence), [0, 1]))
				else:
					test.append((process_sentence(clean_sentence), [0, 1]))

				i+=1

		return train, test, 2

	else:
		raise ValueError("Unknown dataset.")