from ConfigParser import ConfigParser
from math import pow

def iter_list(cfg):

    method = cfg.get('configs', 'method')
    feature = cfg.get('configs', 'feature')

    c_iteration = cfg.getboolean('svm', 'c_iteration')
    def_c_value = cfg.getfloat('svm', 'def_c_value')

    m_list = method.split(',')
    m_list = [i.strip() for i in m_list]
    
    f_list = feature.split(',')
    f_list = [i.strip() for i in f_list]

    i_list = []

    for i in m_list:
        if c_iteration is True:
            for j in range(-4,5):
                for k in f_list:
                    i_list.append({
                            'method': i,
                            'c_value': pow(10, j),
                            'feature': k
                        })
        else:
            for i in f_list:
                i_list.append({
                        'method': i,
                        'c_value': def_c_value,
                        'feature': i
                    })

    print i_list
    return i_list

