from ConfigParser import ConfigParser

class Config():

	cfg = ConfigParser()
	cfg.read('config.ini')

	def get_indexes(self, kind):
		train_str = self.cfg.get('indexes', kind)
		indexes = eval(train_str)
		result = []
		for i in indexes:
			result.extend(range(i[0], i[2], i[1]))
		return sorted(result)

	def get_params(self):

		return {
    		'folder': self.cfg.get('params', 'folder'),
    		'marginX': self.cfg.getint('params', 'marginX'),
    		'marginY': self.cfg.getint('params', 'marginY'),
    		'neg_weight': self.cfg.getint('params', 'neg_weight')
		}

	def get_c_iteration(self):
		return self.cfg.getboolean('svm', 'c_iteration')

	def get_def_c_value(self):
		return self.cfg.getfloat('svm', 'def_c_value')

	def get_methods(self):
		methods = self.cfg.get('configs', 'method')
		m_list = methods.split(',')
		return [i.strip() for i in m_list]

	def get_features(self):
		features = self.cfg.get('configs', 'feature')
		f_list = features.split(',')
		return [i.strip() for i in f_list]

	def get_mode(self):
		return self.cfg.get('configs', 'mode')

	def get_sliding_windows(self):
		return eval(self.cfg.get('svm', 's_windows'))
