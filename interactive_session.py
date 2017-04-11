import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import data
import tensorflow as tf

tf.flags.DEFINE_string("DATASET", None, "Dataset to perform training and testing on")
tf.flags.DEFINE_string("MODEL", None, "Neural network model to use")
tf.flags.DEFINE_string("MODEL_NAME", None, "Name of a saved model")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if FLAGS.DATASET==None:
	raise ValueError("Dataset not specified!")
elif FLAGS.MODEL==None:
	raise ValueError("Neural network model not specified!")
elif FLAGS.MODEL_NAME==None:
	raise ValueError("Model name not specified!")

train, test, num_classes, class_dict, max_sentence_length = data.load_dataset(FLAGS.DATASET)

print("Classes:")
for c in class_dict:
	print(c)

reverse_class_dict=dict()
for k in class_dict.keys():
	reverse_class_dict[tuple(class_dict[k])]=k

with tf.Session() as sess:

	neural_network = data.load_model(sess, model=FLAGS.MODEL, model_name=FLAGS.MODEL_NAME)

	while True:
		sentence = input("> ")
		word_indices = data.index_and_align(data.process_sentence(sentence), max_sentence_length)
		_, prediction = neural_network.feed([word_indices])
		index = int(prediction[0])
		one_hot=tuple([0 if i!=index else 1 for i in range(num_classes)])
		print(reverse_class_dict[one_hot])