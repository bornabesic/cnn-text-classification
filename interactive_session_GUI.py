#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import data
import tensorflow as tf

import tkinter as tk
import tkinter.font

tf.flags.DEFINE_string("MODEL_NAME", None, "Name of a saved model")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if FLAGS.MODEL_NAME==None:
	raise ValueError("Model name not specified!")

tokens = FLAGS.MODEL_NAME.split("_")
dataset = tokens[0]
model = "_".join(["CNN"]+tokens[1:-2])

with tf.Session() as sess:

	neural_network = data.load_model(sess, model=model, model_name=FLAGS.MODEL_NAME)
	input_x = neural_network.input_x
	input_y = neural_network.input_y

	max_sentence_length = input_x.get_shape()[1]
	num_classes = input_y.get_shape()[1]

	class_dict=data.get_dataset_classes(dataset)

	print("Classes:")
	for c in class_dict:
		print(c)
	print()

	reverse_class_dict=dict()
	for k in class_dict.keys():
		reverse_class_dict[tuple(class_dict[k])]=k

	root = tk.Tk()
	root.title("Interactive session")

	font = tk.font.Font(family="Helvetica",size=18,weight="normal")

	input_textbox = tk.Text(root)
	input_textbox.config(font=font)
	probabilities_label = tk.Message(root, text="", width=500)

	def input_changed(event):
		sentence = input_textbox.get("1.0", tk.END).strip()

		word_indices = data.index_and_align(data.process_sentence(sentence), max_sentence_length)
		output, prediction = neural_network.feed([word_indices])
		probabilities = sess.run(tf.nn.softmax(output))[0]

		max_index = int(prediction[0])

		new_text=""
		for i in range(num_classes):
			one_hot=tuple([0 if j!=i else 1 for j in range(num_classes)])
			new_text += "{:10}{:10.2f} % {}".format(reverse_class_dict[one_hot], probabilities[i]*100, "*" if i==max_index else "") + "\n"
		probabilities_label.config(text=new_text.strip())

	def select_all(event):
		event.widget.tag_add("sel","1.0","end")
		return 'break'

	input_textbox.bind("<Control-a>", select_all)
	input_textbox.bind("<KeyRelease>", input_changed)
	input_textbox.grid(row=0, column=0, sticky="wens")
	probabilities_label.grid(row=1, column=0, sticky="wens")
	probabilities_label.config(font=font)

	root.grid_columnconfigure(0,weight=1)
	root.grid_rowconfigure(0,weight=1)
	root.grid_rowconfigure(1,weight=1)

	root.mainloop()
