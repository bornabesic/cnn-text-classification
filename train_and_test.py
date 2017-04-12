#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import data
import time
from datetime import datetime
from word2vec import vector_dimension, embeddings, vocabulary_size
from Logger import Logger
import tensorflow as tf
import numpy as np
import random

tf.flags.DEFINE_integer("BATCH_SIZE", 16, "Training batch size")
tf.flags.DEFINE_integer("NUM_EPOCHS", 150, "Number of training epochs")
tf.flags.DEFINE_string("DATASET", "TREC", "Dataset to perform training and testing on")
tf.flags.DEFINE_string("REGION_SIZES", "5,7", "Region sizes for convolutional layer")
tf.flags.DEFINE_integer("NUM_FILTERS", 64, "Number of filters per region size")
tf.flags.DEFINE_boolean("STATIC_EMBEDDINGS", True, "Word2Vec embeddings will not be fine-tuned during the training")
tf.flags.DEFINE_float("REG_LAMBDA", 0.6, "Lambda regularization parameter")
tf.flags.DEFINE_float("DROPOUT_KEEP_PROB", 0.5, "Neuron keep probability for dropout layer")
tf.flags.DEFINE_string("MODEL", "SentenceCNN", "Neural network model to use")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

today = datetime.today()

id_string = "{}-{}-{}-{}-{}-{}-{}".format(FLAGS.DATASET, today.day, today.month, today.year, today.hour, today.minute, today.second)

logger = Logger(
	id_string+".txt",
	print_to_stdout=True
)

logger.log("Hyperparameters:")
for param, value in FLAGS.__flags.items():
	logger.log(param + ": " + str(value))
logger.log()

train, test, num_classes, class_dict, max_sentence_length = data.load_dataset(FLAGS.DATASET)

logger.log("Train set size: " + str(len(train)))
logger.log("Test set size: " + str(len(test)))
logger.log("Classes: " + str(num_classes))
logger.log("Max sentence length: " + str(max_sentence_length))
logger.log()

# train data prepare

for i in range(len(train)):
	sentence, label = train[i]
	word_indices = data.index_and_align(sentence, max_sentence_length)
	train[i]=(word_indices,label)

# test data prepare

for i in range(len(test)):
	sentence, label = test[i]
	word_indices = data.index_and_align(sentence, max_sentence_length)
	test[i]=(word_indices,label)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess, logger:

	model_class = data.get_model_class(FLAGS.MODEL)

	neural_network = model_class(
		model_name=id_string,
		session=sess,
		learning_rate=3e-4,
		optimizer=tf.train.AdamOptimizer,
		filter_sizes=[int(region_size) for region_size in FLAGS.REGION_SIZES.split(",")],
		num_filters=FLAGS.NUM_FILTERS,
		embeddings=embeddings,
		vocabulary_size=vocabulary_size,
		static=FLAGS.STATIC_EMBEDDINGS,
		max_sentence_length=max_sentence_length,
		num_classes=num_classes,
		embedding_dim=vector_dimension,
		regularization_lambda=FLAGS.REG_LAMBDA,
		dropout_keep_prob=FLAGS.DROPOUT_KEEP_PROB
	)

	start_time = time.time()
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
			logger.log("Epoch " + str(epoch) + " loss: " + str(avg_loss))

	except KeyboardInterrupt:
		pass

	end_time=time.time()
	training_minutes=int((end_time-start_time)//60)
	training_seconds=int((end_time-start_time)-training_minutes*60)

	logger.log("Training DONE ({} m {} s). Evaluating...".format(training_minutes, training_seconds))
	correct=0
	for i in range(len(test)):
		indices, label = test[i]
		output, predictions = neural_network.feed([indices])
		accuracy=label[predictions[0]]
		correct+=accuracy

	logger.log("Test set accuracy: " + str(correct/len(test)*100) + " %")

	data.save_model(neural_network)
