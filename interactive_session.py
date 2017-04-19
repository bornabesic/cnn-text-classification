#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import data
import tensorflow as tf

tf.flags.DEFINE_string("MODEL", None, "Neural network model to use")
tf.flags.DEFINE_string("MODEL_NAME", None, "Name of a saved model")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if FLAGS.MODEL==None:
	raise ValueError("Neural network model not specified!")
elif FLAGS.MODEL_NAME==None:
	raise ValueError("Model name not specified!")


with tf.Session() as sess:

	neural_network = data.load_model(sess, model=FLAGS.MODEL, model_name=FLAGS.MODEL_NAME)
	input_x = neural_network.input_x
	input_y = neural_network.input_y

	max_sentence_length = input_x.get_shape()[1]
	num_classes = input_y.get_shape()[1]

	dataset = FLAGS.MODEL_NAME.split("-")[0]

	class_dict=data.get_dataset_classes(dataset)

	print("Classes:")
	for c in class_dict:
		print(c)

	reverse_class_dict=dict()
	for k in class_dict.keys():
		reverse_class_dict[tuple(class_dict[k])]=k

	while True:
		sentence = input("> ")
		word_indices = data.index_and_align(data.process_sentence(sentence), max_sentence_length)
		_, prediction = neural_network.feed([word_indices])
		index = int(prediction[0])
		one_hot=tuple([0 if i!=index else 1 for i in range(num_classes)])
		print(reverse_class_dict[one_hot])
