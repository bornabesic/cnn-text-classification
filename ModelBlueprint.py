import tensorflow as tf

class ModelBlueprint:

	def __init__(self,
		model_name=None, session=None,
		learning_rate=None, optimizer=None,
	):

		if model_name==None:
			return

		self.model_name=model_name
		self.session=session
		self.learning_rate=learning_rate
		self.optimizer=optimizer

		###############
		#
		#	TODO model definition
		#	self.input_x	name="input_x"
		#	self.input_y	name="input_y"
		#	self.loss		name="loss"
		#	self.output		name="output"
		#	(op name has to be same as variable name)
		#	+ additional parameters


				# TODO code here

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
		_, loss = self.session.run([self.train_op, self.loss], feed_dict={self.input_x: input_x, self.input_y: input_y}) # TODO additional parameters
		return loss

	def feed(self, input_x): # TODO additional parameters
		return self.session.run([self.output], feed_dict={self.input_x: input_x}) # TODO additional parameters