import tensorflow as tf

class CNN_NonLinearFC:

	def __init__(self,
		model_name=None, session=None,
		learning_rate=None, optimizer=None,
		learning_decay_rate=None,
		filter_sizes=None,
		num_filters=None,
		max_sentence_length=None,
		num_classes=None,
		embeddings=None,
		new_embeddings=None,
		embedding_dim=None,
		vocabulary_size=None,
		static=None,
		max_l2_norm=None,
		regularization_lambda=None,
		dropout_keep_prob=None
	):

		if model_name==None:
			return

		self.model_name=model_name
		self.session=session
		self.learning_rate=learning_rate
		self.optimizer=optimizer
		self.dropout_keep_prob_train=dropout_keep_prob
		self.regularization_lambda=regularization_lambda


		###############
		#
		#	model definition


		self.input_x = tf.placeholder(shape=(None, max_sentence_length), dtype=tf.int32, name="input_x")
		self.input_y = tf.placeholder(shape=(None, num_classes), dtype=tf.float32, name="input_y")
		self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")

		# ===== EMBEDDING LAYER
		self.embeddings_placeholder = tf.placeholder(tf.float32, shape=(vocabulary_size, embedding_dim))

		self.embeddings=tf.Variable(self.embeddings_placeholder, trainable = not static)
		self.new_embeddings=tf.Variable(new_embeddings, trainable = True)

		self.all_embeddings = tf.concat([self.embeddings, self.new_embeddings], axis=0)

		self.embedded_words = tf.nn.embedding_lookup(self.all_embeddings, self.input_x)

		# ===== CONVOLUTIONAL LAYER
		self.input_x_expanded = tf.expand_dims(self.embedded_words, -1)

		self.pool_results=[]
		for i, filter_size in enumerate(filter_sizes):

			filter = tf.get_variable("filter"+str(i), shape=(filter_size, embedding_dim, 1, num_filters), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			if max_l2_norm!=0:
				filter = tf.clip_by_norm(filter, max_l2_norm)
			bias = tf.Variable(tf.constant(0.0, shape=(num_filters,)))

			conv = tf.nn.conv2d(
				input=self.input_x_expanded, # [batch, in_height, in_width, in_channels]
				filter=filter, # [filter_height, filter_width, in_channels, out_channels]
				strides=[1, 1, 1, 1],
				padding="VALID"
			)

			relu = tf.nn.relu(tf.nn.bias_add(conv, bias))

			conv_dim = max_sentence_length-filter_size+1

			pooled = tf.nn.max_pool(
				relu,
				ksize=[1, conv_dim, 1, 1],
				strides=[1, 1, 1, 1],
				padding='VALID'
			)
			self.pool_results.append(pooled)

		# FLATTENING LAYER

		num_filters_total = num_filters * len(filter_sizes)
		self.flat = tf.reshape(tf.concat(self.pool_results, 3), [-1, num_filters_total])

		# DROPOUT LAYER

		self.dropout = tf.nn.dropout(self.flat, self.dropout_keep_prob)

		# FULLY CONNECTED LAYERS

		# HIDDEN LAYER

		hidden_layer_size = num_classes
		W_hidden = tf.get_variable("W_hidden", shape=(num_filters_total, hidden_layer_size), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_hidden = tf.Variable(tf.constant(0.1, shape=(hidden_layer_size,)))

		out_hidden = tf.nn.relu(tf.nn.xw_plus_b(self.dropout, W_hidden, b_hidden))

		# OUTPUT LAYER

		W_output = tf.get_variable("W_output", shape=(hidden_layer_size, num_classes), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_output = tf.Variable(tf.constant(0.1, shape=(num_classes,)))

		self.output = tf.nn.xw_plus_b(out_hidden, W_output, b_output, name="output")

		self.predictions = tf.argmax(self.output, 1, name="predictions")

		losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.output)
		if regularization_lambda!=0:
			l2_loss = tf.nn.l2_loss(W_hidden)+tf.nn.l2_loss(W_output)
			self.loss = tf.reduce_mean(tf.add(losses, tf.multiply(self.regularization_lambda, l2_loss)), name="loss")
		else:
			self.loss = tf.reduce_mean(losses, name="loss")


		#
		#
		###############

		global_step = tf.Variable(0, trainable=False)

		# optimization method
		self.optimizer = optimizer(
			learning_rate=tf.train.exponential_decay(learning_rate, global_step, 1000000, learning_decay_rate, staircase=True)
		)

		# training operation
		self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

		# saver
		self.saver = tf.train.Saver()

		# initialize variables
		self.session.run(tf.global_variables_initializer(), feed_dict={self.embeddings_placeholder: embeddings})

	def train_step(self, input_x, input_y):
		_, loss = self.session.run([self.train_op, self.loss], feed_dict={self.input_x: input_x, self.input_y: input_y, self.dropout_keep_prob: self.dropout_keep_prob_train})
		return loss

	def feed(self, input_x):
		return self.session.run([self.output, self.predictions], feed_dict={self.input_x: input_x, self.dropout_keep_prob: 1})
