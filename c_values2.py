from engine.parts import train, test, bootstrap, compute_BG_Image, train_svm, sw_search
import numpy as np
from math import pow
from config import Config
from utils.iteration_manager import iter_list



cfg = Config()

train_indexes = cfg.get_indexes('train')
bootstrap_indexes = cfg.get_indexes('bootstrap')
test_indexes = cfg.get_indexes('test')

params = cfg.get_params()
mode = cfg.get_mode()

i_list = iter_list()
iter_count = len(i_list)

print str(iter_count) + " iterations to go..."

index = 1
for i in i_list:

    print str(index) + "th iteration of " + str(iter_count) + " iterations."
    index += 1

    method = str(i['method'])
    c_value = str(i['c_value'])
    feature = str(i['feature'])
    params['method'] = method
    params['feature'] = feature

    print "method: " + method + ", c_value: " + c_value + ", feature: " + feature

    BG_img = compute_BG_Image(params['folder'], [1,2,3,4,5,6,7])

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
