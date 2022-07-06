import logging

from .TRN import TRN_base, TRN_fcbn, TRN_cls
from .ATCoN import ATCoN_base, ATCoN_fcbn, ATCoN_cls
from .config import get_config

def get_target_symbol(name, print_net=False, segments=5, consensus_type='avg', fcbn_type='ori', classifier_type='ori', batch_size=16, **kwargs):

	logging.info("Network:: Getting symbol using {} network.".format(name))

	if name.upper() == 'TRN':
		logging.info("Network:: For frame-based method using {} segments".format(segments))
		net_feat = TRN_base(segments=segments, consensus_type=consensus_type, **kwargs)
		net_fc = TRN_fcbn(consensus_type=consensus_type, fcbn_type=fcbn_type, **kwargs)
		net_cls = TRN_cls(consensus_type=consensus_type, classifier_type=classifier_type, **kwargs)
		network = [net_feat, net_fc, net_cls]
	elif name.upper() == 'ATCON':
		logging.info("Network:: For frame-based method using {} segments".format(segments))
		net_feat = ATCoN_base(segments=segments, consensus_type=consensus_type, **kwargs)
		net_fc = ATCoN_fcbn(consensus_type=consensus_type, fcbn_type=fcbn_type, **kwargs)
		net_cls = ATCoN_cls(consensus_type=consensus_type, classifier_type=classifier_type, **kwargs)
		network = [net_feat, net_fc, net_cls]

	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, **kwargs)
	return network, input_conf
