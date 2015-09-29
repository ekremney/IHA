from engine.parts import train, test, bootstrap, compute_BG_Image, train_svm, sw_search
import numpy as np
from math import pow
from ConfigParser import ConfigParser

# Reads configuration values from config.ini file
cfg = ConfigParser()
cfg.read('config.ini')

method = cfg.get('configs', 'method')
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
    'neg_weight': cfg.getint('params', 'neg_weight'),
    'method': method
}

c_iteration = cfg.getboolean('svm', 'c_iteration')
def_c_value = cfg.getfloat('svm', 'def_c_value')

# Opens a text file to note test results
text_file = open("output2.txt", "a")
text_file.write("Method used: " + params['method'] + "\n")
text_file.write("-------------------------------------\n")


# If c_iteration is True, iterates from 0.0001 to 10000 for c values.
# Else, uses default c value stated in config.ini file
if c_iteration is True:
    for i in range(-4, 5):
        BG_img = compute_BG_Image(params['folder'], train_indexes)
        trf, trl, trfc = train(train_indexes, BG_img, params)
        svm_param = ' -s 0 -t 0 -c ' + str(pow(10, i))
        print "c value is: " + str(pow(10, i))
        svm = train_svm(trf, trl, svm_param)
        trf2, trl2 = bootstrap(bootstrap_indexes, BG_img, params, trf, trl, trfc, svm)
        svm2 = train_svm(trf2, trl2, svm_param)
        svm_AP, svm_PR, svm_RC = sw_search(test_indexes, BG_img, params, svm2)
        ap_mean = np.mean(svm_AP)
        text_file.write("precision for c value " + str(pow(10, i)) + " is: " + str(ap_mean) + "\n")
else:
    BG_img = compute_BG_Image(params['folder'], train_indexes)

    trf, trl, trfc = train(train_indexes, BG_img, params)

    svm_param = ' -s 0 -t 0 -c ' + str(def_c_value)
    svm = train_svm(trf, trl, svm_param)

    trf2, trl2 = bootstrap(bootstrap_indexes, BG_img, params, trf, trl, trfc, svm)

    svm2 = train_svm(trf2, trl2, svm_param)
    svm_AP, svm_PR, svm_RC = sw_search(test_indexes, BG_img, params, svm2)
    ap_mean = np.mean(svm_AP)
    text_file.write("precision for c value " + str(def_c_value) + " is: " + str(ap_mean) + "\n")



# Closes the text file
text_file.write("*************************************\n\n\n")
text_file.close()
