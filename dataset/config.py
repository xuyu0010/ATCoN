import logging

def get_config(name):

	config = {}

	if name.upper() == 'DAILY':
		config['num_classes'] = 8
	elif name.upper() == 'UH':
		config['num_classes'] = 12
	elif name.upper() == 'SPORTS':
		config['num_classes'] = 23
	else:
		logging.error("Configs for dataset '{}' not found".format(name))
		raise NotImplemented

	logging.debug("Dataset: '{}', configs: {}".format(name.upper(), config))

	return config


if __name__ == "__main__":
	logging.getLogger().setLevel(logging.DEBUG)

	logging.info(get_config("DAILY"))
