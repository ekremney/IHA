from config import Config
from math import pow

def iter_list():

    cfg = Config()

    methods = cfg.get_methods()
    features = cfg.get_features()
    c_iteration = cfg.get_c_iteration()
    def_c_value = cfg.get_def_c_value()

    i_list = []

    for i in methods:
        if c_iteration is True:
            for j in range(-4,5):
                for k in features:
                    i_list.append({
                            'method': i,
                            'c_value': pow(10, j),
                            'feature': k
                        })
        else:
            for j in features:
                i_list.append({
                        'method': i,
                        'c_value': def_c_value,
                        'feature': j
                    })
    return i_list

