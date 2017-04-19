import struct
import numpy as np

indices_dictionary=dict()
vocabulary_size=300000

vector_dimension=300
vector_size = vector_dimension*4

vector_struct = struct.Struct("300f")

embeddings=np.empty(shape=[vocabulary_size, vector_dimension], dtype=np.float32)
new_embeddings = np.empty(shape=[0, vector_dimension], dtype=np.float32)
new_index=0

# a = sqrt(3*Var(X))
# a = stddev(X) * sqrt(3)
uniform_a = np.std(embeddings,0) * np.sqrt(3)

# generate new vector with same variance as existing ones
def generate_new_vector():
	global new_index

	new_embeddings.resize([new_index+1, vector_dimension], refcheck=False)
	new_embeddings[new_index]=[np.random.uniform(-a, a) for a in uniform_a]
	new_index+=1

	return vocabulary_size+new_index-1


def word_index(word, generate=False):
	if word not in indices_dictionary:
		if not generate:
			return 0

		index = generate_new_vector()
		indices_dictionary[word]=index

	return indices_dictionary[word]

def load_embeddings():
	with open("word2vec_vectors", "rb") as word2vec_vectors_file:
		for i in range(vocabulary_size):
			bytes = word2vec_vectors_file.read(vector_size)
			embeddings[i]=vector_struct.unpack(bytes)

	# vector for unknown words
	embeddings[0]=[0 for _ in range(vector_dimension)]

def load_new_indices(file_path):
	with open(file_path, "r", encoding="utf8") as new_indices_file:
		for line in new_indices_file:
			line_tokens = line.strip("\n").split("\t")
			indices_dictionary[line_tokens[0]]=int(line_tokens[1])

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
