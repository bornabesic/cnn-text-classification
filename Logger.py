from os import mkdir

class Logger:
	def __init__(self, file_name, print_to_stdout=False):
		base_path="./logs/"
		try:
			mkdir(base_path)
		except FileExistsError:
			pass

		self.print_to_stdout=print_to_stdout
		self.file = open(base_path+file_name, "w", encoding="UTF-8")

	def log(self, message=""):
		if self.print_to_stdout:
			print(message)
		print(message, file=self.file)

	def close(self):
		self.file.close()
