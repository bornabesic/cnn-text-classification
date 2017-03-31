import struct
import numpy as np

indices_dictionary=dict()
vocabulary_size=300000 # maximum

vector_dimension=300
vector_size = vector_dimension*4

vector_struct = struct.Struct("300f")

embeddings=np.empty(shape=[vocabulary_size, vector_dimension], dtype=np.float32)

with open("word2vec_vectors", "rb") as word2vec_vectors_file:
	for i in range(vocabulary_size):
		bytes = word2vec_vectors_file.read(vector_size)
		embeddings[i]=vector_struct.unpack(bytes)

# TODO vector for unknown words
# embeddings[0]=[0 for _ in range(vector_dimension)]
#

# initialization
with open("word2vec_indices", "r", encoding="utf8") as word2vec_indices_file:
	word_counter=0
	for line in word2vec_indices_file:
		if word_counter<vocabulary_size:
			line_tokens = line.strip("\n").split("\t")
			indices_dictionary[line_tokens[0]]=int(line_tokens[1])
		else:
			break
		word_counter+=1

def word_index(word):
	if word not in indices_dictionary:
		return 0
	else:
		return indices_dictionary[word]
