import logging

def get_config(name, **kwargs):

    logging.debug("loading network configs of: {}".format(name.upper()))

    config = {}

    logging.info("Preprocessing:: using Video default mean & std.")
    config['mean'] = [124 / 255, 117 / 255, 104 / 255]
    config['std'] = [1 / (.0167 * 255)] * 3

    logging.info("data:: {}".format(config))
    return config