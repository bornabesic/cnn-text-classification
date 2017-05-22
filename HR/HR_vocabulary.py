import struct
import numpy as np

indices_dictionary=dict()

def word_index(word):
	if word not in indices_dictionary:
		return 0

	return indices_dictionary[word]

def load_indices(most_frequent_words_list):
	i=0
	for word in most_frequent_words_list:
			indices_dictionary[word]=i
			i+=1

def load_new_indices(file_path):
	with open(file_path, "r", encoding="utf8") as new_indices_file:
		for line in new_indices_file:
			line_tokens = line.strip("\n").split("\t")
			indices_dictionary[line_tokens[0]]=int(line_tokens[1])
