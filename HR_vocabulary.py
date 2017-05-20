import struct
import numpy as np

indices_dictionary=dict()
vocabulary_size=0

def word_index(word):
	if word not in indices_dictionary:
		return 0

	return indices_dictionary[word]

def load_indices(dataset_name):
	global vocabulary_size

	#indices_dictionary=dict()
	with open("./HR_datasets/"+dataset_name+"/indices", "r", encoding="utf-8") as indices_file:
		for line in indices_file:
			line_tokens = line.strip("\n").split("\t")
			indices_dictionary[line_tokens[0]]=int(line_tokens[1])
			vocabulary_size+=1

def load_new_indices(file_path):
	with open(file_path, "r", encoding="utf8") as new_indices_file:
		for line in new_indices_file:
			line_tokens = line.strip("\n").split("\t")
			indices_dictionary[line_tokens[0]]=int(line_tokens[1])
