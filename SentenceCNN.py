import tensorflow as tf

class SentenceCNN:

	def __init__(self,
		model_name, session,
		learning_rate, optimizer,
		filter_sizes,
		num_filters,
		max_sentence_length,
		num_classes,
		embeddings,
		embedding_dim,
		vocabulary_size,
		static,
		regularization_lambda,
		dropout_keep_prob
	):
		self.model_name=model_name
		self.session=session
		self.learning_rate=learning_rate
		self.optimizer=optimizer
		self.dropout_keep_prob_train=dropout_keep_prob
		self.regularization_lambda=regularization_lambda

		self.base_path = "./"+model_name+"/"


		###############
		#
		#	TODO model definition
		#	self.input_x	name="input_x"
		#	self.input_y	name="input_y"
		#	self.loss		name="loss"
		#	self.output		name="output"
		#	+ additional parameters

		self.input_x = tf.placeholder(shape=[None, max_sentence_length], dtype=tf.int32, name="input_x")
		self.input_y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32, name="input_y")
		self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")

		# ===== EMBEDDING LAYER
		self.embeddings_placeholder = tf.placeholder(tf.float32, shape=(vocabulary_size, embedding_dim))

		self.embeddings=tf.Variable(self.embeddings_placeholder, trainable = not static)
		self.embedded_words = tf.nn.embedding_lookup(self.embeddings, self.input_x)

		# ===== CONVOLUTIONAL LAYER
		self.input_x_expanded = tf.expand_dims(self.embedded_words, -1)

		self.pool_results=[]
		for filter_size in filter_sizes:


			filter = tf.Variable(tf.truncated_normal(shape=[filter_size, embedding_dim, 1, num_filters], mean=0.5, stddev=0.2))
			#filter = tf.Variable(tf.truncated_normal(shape=[filter_size, embedding_dim, 1, num_filters]))
			bias = tf.Variable(tf.constant(0.0, shape=[num_filters]))

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

		# FULLY CONNECTED LAYER

		W = tf.Variable(tf.truncated_normal(shape=[num_filters_total, num_classes]))
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]))

		l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

		self.output = tf.nn.xw_plus_b(self.dropout, W, b)
		self.predictions = tf.argmax(self.output, 1)


		losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.output)
		self.loss = tf.reduce_mean(losses) + self.regularization_lambda * l2_loss



		#
		#
		###############

		# optimization method
		self.optimizer = optimizer(learning_rate=self.learning_rate)

		# training operation
		self.train_op = self.optimizer.minimize(self.loss)

		# saver
		self.saver = tf.train.Saver()

		# initialize variables
		self.session.run(tf.global_variables_initializer(), feed_dict={self.embeddings_placeholder: embeddings})

	def train_step(self, input_x, input_y): # TODO additional parameters
		_, loss = self.session.run([self.train_op, self.loss], feed_dict={self.input_x: input_x, self.input_y: input_y, self.dropout_keep_prob: self.dropout_keep_prob_train}) # TODO additional parameters
		return loss

	def feed(self, input_x): # TODO additional parameters
		return self.session.run([self.output, self.predictions], feed_dict={self.input_x: input_x, self.dropout_keep_prob: 1}) # TODO additional parameters

	def save(self, step=None):
		from os import mkdir

		try:
			mkdir(self.base_path)
		except FileExistsError:
			pass

		self.saver.save(sess=self.session, save_path=self.base_path+self.model_name, global_step=step)

	def load(self):
		latest = tf.train.latest_checkpoint(self.base_path)

		self.saver = tf.train.import_meta_graph(latest+".meta")
		self.saver.restore(self.session, latest)

		self.input_x = self.session.graph.get_operation_by_name("input_x").outputs[0]
		self.input_y = self.session.graph.get_operation_by_name("input_y").outputs[0]
		self.output = self.session.graph.get_operation_by_name("output").outputs[0]
		self.loss = self.session.graph.get_operation_by_name("loss").outputs[0]
		# TODO + additional parameters
