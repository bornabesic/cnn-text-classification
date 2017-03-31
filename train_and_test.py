#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import data
from word2vec import word_index, vector_dimension, embeddings, vocabulary_size
from SentenceCNN import SentenceCNN
import tensorflow as tf
import numpy as np
import random

tf.flags.DEFINE_integer("BATCH_SIZE", 64, "Training batch size")
tf.flags.DEFINE_integer("NUM_EPOCHS", 150, "Number of training epochs")
tf.flags.DEFINE_string("DATASET", "TREC", "Dataset to perform training and testing on")
tf.flags.DEFINE_integer("NUM_FILTERS", 64, "Number of filters per region size")
tf.flags.DEFINE_boolean("STATIC_EMBEDDINGS", True, "Word2Vec embeddings will not be fine-tuned during the training")

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

# train data prepare

for i in range(len(train)):
	sentence, label = train[i]
	words = sentence.split(" ")

	pad_size = max_sentence_length-len(words)

	word_indices = [word_index(w) for w in words] + [0 for _ in range(pad_size)]
	train[i]=(word_indices,label)

# test data prepare

for i in range(len(test)):
	sentence, label = test[i]
	words = sentence.split(" ")

	pad_size = max_sentence_length-len(words)

	word_indices = [word_index(w) for w in words] + [0 for _ in range(pad_size)]
	test[i]=(word_indices,label)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:

	#TREC filter_sizes=[5, 7], num_filters=64

	neural_network = SentenceCNN(
		model_name="SentenceClassifier",
		session=sess,
		learning_rate=3e-4,
		optimizer=tf.train.AdamOptimizer,
		filter_sizes=[5, 7],
		num_filters=FLAGS.NUM_FILTERS,
		embeddings=embeddings,
        vocabulary_size=vocabulary_size,
		static=FLAGS.STATIC_EMBEDDINGS,
		max_sentence_length=max_sentence_length,
		num_classes=num_classes,
		embedding_dim=vector_dimension,
		regularization_lambda=0.9,
		dropout_keep_prob=0.6
	)

	try: # allow user to end training using Ctrl+C
		for epoch in range(FLAGS.NUM_EPOCHS):
			random.shuffle(train)
			avg_loss=0

			i=0
			while i<len(train)-FLAGS.BATCH_SIZE:
				indices, labels = zip(*train[i:i+FLAGS.BATCH_SIZE])
				loss = neural_network.train_step(indices, labels)
				avg_loss+=loss
				i+=FLAGS.BATCH_SIZE
			avg_loss/=(i/FLAGS.BATCH_SIZE)
			print("Epoch " + str(epoch) + " loss: " + str(avg_loss))

	except KeyboardInterrupt:
		pass

	print("Training DONE. Evaluating...")
	correct=0
	for i in range(len(test)):
		indices, label = test[i]
		output, predictions = neural_network.feed([indices])
		accuracy=label[predictions[0]]
		correct+=accuracy

	print("Test set accuracy: " + str(correct/len(test)*100) + " %")
