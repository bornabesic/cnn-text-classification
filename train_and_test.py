import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import data
from word2vec import get_embedding_vector, vector_dimension
from SentenceCNN import SentenceCNN
import tensorflow as tf
import numpy as np
import random

tf.flags.DEFINE_integer("BATCH_SIZE", 16, "Training batch size")
tf.flags.DEFINE_integer("NUM_EPOCHS", 150, "Number of training epochs")
tf.flags.DEFINE_string("DATASET", "TREC", "Dataset to perform training and testing on")
tf.flags.DEFINE_integer("NUM_FILTERS", 64, "Number of filters per region size")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("Hyperparameters:")
for param, value in FLAGS.__flags.items():
    print("{}={}".format(param, value))
print("")

train, test, num_classes = data.load(FLAGS.DATASET)

max_sentence_length=0
for sentence, label in train:
	tokens = sentence.split(" ")
	if len(tokens)>max_sentence_length:
		max_sentence_length=len(tokens)

print("Max sentence length: " + str(max_sentence_length))

padding_vector = get_embedding_vector("__PAD__")

# train data prepare

for i in range(len(train)):
	sentence, label = train[i]
	words = sentence.split(" ")
	embeddings = [get_embedding_vector(w) for w in words]  + [padding_vector for _ in range(max_sentence_length-len(words))]

	train[i] = (embeddings, label)

# test data prepare

for i in range(len(test)):
	sentence, label = test[i]
	words = sentence.split(" ")
	embeddings = [get_embedding_vector(w) for w in words]  + [padding_vector for _ in range(max_sentence_length-len(words))]

	test[i] = (embeddings, label)

with tf.Session() as sess:

	#TREC filter_sizes=[5, 7], num_filters=64

	neural_network = SentenceCNN(
		model_name="SentenceClassifier",
		session=sess,
		learning_rate=3e-4,
		optimizer=tf.train.AdamOptimizer,
		filter_sizes=[5, 7],
		num_filters=FLAGS.NUM_FILTERS,
		max_sentence_length=max_sentence_length,
		num_classes=num_classes,
		embedding_dim=vector_dimension,
		regularization_lambda=0.8,
		dropout_keep_prob=0.6
	)

	try: # allow user to end training using Ctrl+C
		for epoch in range(FLAGS.NUM_EPOCHS):
			random.shuffle(train)
			avg_loss=0

			i=0
			while i<len(train)-FLAGS.BATCH_SIZE:
				embeddings, labels = zip(*train[i:i+FLAGS.BATCH_SIZE])
				loss = neural_network.train_step(embeddings, labels)
				avg_loss+=loss
				i+=FLAGS.BATCH_SIZE
			avg_loss/=(i/FLAGS.BATCH_SIZE)
			print("Epoch " + str(epoch) + " loss: " + str(avg_loss))

	except KeyboardInterrupt:
		pass

	print("Training DONE. Evaluating...")
	correct=0
	for i in range(len(test)):
		embedding, label = test[i]
		output, predictions = neural_network.feed([embedding])
		accuracy=label[predictions[0]]
		correct+=accuracy
	
	print("Test set accuracy: " + str(correct/len(test)*100) + " %")