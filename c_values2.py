from engine.parts import train, test, bootstrap, compute_BG_Image, train_svm, sw_search
import numpy as np
from math import pow
from ConfigParser import ConfigParser
from utils.iteration_manager import iter_list

# Reads configuration values from config.ini file
cfg = ConfigParser()
cfg.read('config.ini')

mode = cfg.get('configs', 'mode')

train_indexes = range(cfg.getint('train_indexes', 'start'), cfg.getint('train_indexes', 'end'),
                      cfg.getint('train_indexes', 'step'))
bootstrap_indexes = range(cfg.getint('bootstrap_indexes', 'start'), cfg.getint('bootstrap_indexes', 'end'),
                          cfg.getint('bootstrap_indexes', 'step'))
test_indexes = range(cfg.getint('test_indexes', 'start'), cfg.getint('test_indexes', 'end'),
                     cfg.getint('test_indexes', 'step'))

params = {
    'folder': cfg.get('params', 'folder'),
    'marginX': cfg.getint('params', 'marginX'),
    'marginY': cfg.getint('params', 'marginY'),
    'neg_weight': cfg.getint('params', 'neg_weight')
}

i_list = iter_list(cfg)
iter_count = len(i_list)
print str(iter_count) + " iterations to go..."
index = 0
for i in i_list:

    print str(index) + "th iteration of " + str(iter_count) + " iterations."
    index += 1

    method = str(i['method'])
    c_value = str(i['c_value'])
    feature = str(i['feature'])
    params['method'] = method
    params['feature'] = feature

    print "method: " + method + ", c_value: " + c_value

    BG_img = compute_BG_Image(params['folder'], train_indexes)

    trf, trl, trfc = train(train_indexes, BG_img, params)
    svm_param = ' -s 0 -t 0 -c ' + c_value
    svm = train_svm(trf, trl, svm_param)

    trf2, trl2 = bootstrap(bootstrap_indexes, BG_img, params, trf, trl, trfc, svm)
    svm2 = train_svm(trf2, trl2, svm_param)

    svm_AP, svm_PR, svm_RC = sw_search(test_indexes, BG_img, params, svm2)
    ap_mean = np.mean(svm_AP)

    text_file = open("output2.txt", "a")
    text_file.write("method: "+ method + ", c value: " + c_value + ", feature: " + feature + ", precision: " + str(ap_mean) + "\n")
    text_file.close()
