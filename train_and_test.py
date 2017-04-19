#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import data
import time
import word2vec
import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from Logger import Logger

tf.flags.DEFINE_integer("BATCH_SIZE", 50, "Training batch size")
tf.flags.DEFINE_integer("NUM_EPOCHS", 500, "Number of training epochs")
tf.flags.DEFINE_string("DATASET", "TREC", "Dataset to perform training and testing on")
tf.flags.DEFINE_string("REGION_SIZES", "3,4,5", "Region sizes for convolutional layer")
tf.flags.DEFINE_integer("NUM_FILTERS", 100, "Number of filters per region size")
tf.flags.DEFINE_boolean("STATIC_EMBEDDINGS", True, "Word2Vec embeddings will not be fine-tuned during the training")
tf.flags.DEFINE_float("MAX_L2_NORM", 3, "Maximum L2 norm for convolutional layer weights")
tf.flags.DEFINE_float("REG_LAMBDA", 0, "Lambda regularization parameter")
tf.flags.DEFINE_float("DROPOUT_PROB", 0.3, "Neuron dropout probability")
tf.flags.DEFINE_float("LEARNING_RATE", 1e-3, "Initial learning rate value")
tf.flags.DEFINE_float("LEARNING_DECAY_RATE", 0.95, "Rate at which learning rate will exponentially decay during the training")
tf.flags.DEFINE_string("MODEL", "CNN_YoonKim_Xavier", "Neural network model to use")

tf.flags.DEFINE_boolean("SAVE", False, "Model will be saved")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

today = datetime.today()

id_string = "{}_{}_{:02}-{:02}-{:02}_{:02}-{:02}".format(
	FLAGS.DATASET,
	"_".join(FLAGS.MODEL.split("_")[1:]),
	today.day,
	today.month,
	int(str(today.year)[-2:]),
	today.hour,
	today.minute
)

logger = Logger(
	id_string+".txt",
	print_to_stdout=True
)

logger.log("Hyperparameters:")
for param, value in sorted(FLAGS.__flags.items()):
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
	word_indices = data.index_and_align(sentence, max_sentence_length, generate_new_vector=True)
	train[i]=(word_indices,label)

# test data prepare

for i in range(len(test)):
	sentence, label = test[i]
	word_indices = data.index_and_align(sentence, max_sentence_length, generate_new_vector=False)
	test[i]=(word_indices,label)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess, logger:

	model_class = data.get_model_class(FLAGS.MODEL)

	neural_network = model_class(
		model_name=id_string,
		session=sess,
		learning_rate=FLAGS.LEARNING_RATE,
		learning_decay_rate=FLAGS.LEARNING_DECAY_RATE,
		optimizer=tf.train.AdamOptimizer,
		filter_sizes=[int(region_size) for region_size in FLAGS.REGION_SIZES.split(",")],
		num_filters=FLAGS.NUM_FILTERS,
		embeddings=word2vec.embeddings,
		new_embeddings=word2vec.new_embeddings,
		vocabulary_size=word2vec.vocabulary_size,
		static=FLAGS.STATIC_EMBEDDINGS,
		max_sentence_length=max_sentence_length,
		num_classes=num_classes,
		embedding_dim=word2vec.vector_dimension,
		max_l2_norm=FLAGS.MAX_L2_NORM,
		regularization_lambda=FLAGS.REG_LAMBDA,
		dropout_keep_prob=1-FLAGS.DROPOUT_PROB
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

	if FLAGS.SAVE:
		data.save_model(neural_network)
