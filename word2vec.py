import struct
import numpy as np

# initialization

indices_dictionary=dict()

with open("word2vec_vectors", "rb") as word2vec_vectors_file:
	word2vec_vectors=word2vec_vectors_file.read()

with open("word2vec_indices", "r", encoding="utf8") as word2vec_indices_file:
	for line in word2vec_indices_file:
		line_tokens = line.strip("\n").split("\t")
		indices_dictionary[line_tokens[0]]=int(line_tokens[1])
		del line_tokens

vector_dimension=300
vector_size = vector_dimension*4
vector_struct = struct.Struct("300f")
empty_vector = [0 for _ in range(vector_dimension)]

def get_embedding_vector(word):
	if word=="__PAD__":
		return empty_vector

	try:
		word_index = indices_dictionary[word]
		start = word_index*vector_size
		end = start + vector_size
		return list(vector_struct.unpack(word2vec_vectors[start:end]))

	except KeyError:
		# TODO
		# ako nema rijeci u rjecniku generirati nasumican vektor dimenzije 300 iz U[-a, a]
		# a izabrati tako da je varijanca ista kao kod dosadasnjih poznatih vektora
		#vectors[word]=(np.random.rand(vector_dimension)*1.5-0.75).tolist()
		#return vectors[word]
		return empty_vector


if __name__=="__main__":
	print(get_embedding_vector("</s>"))




