from utils.arrayops import array_in_range
from utils.imageops import img_read, read_motion_image, read_bboxes, add_bbox_margin, img_crop, rand_bbox, overlaps, detect_vehicles, non_max_suppression, compute_detection_AP
from random import sample
from libsvm.svmutil import svm_train, svm_predict
from engine.parts import train, test, bootstrap, compute_BG_Image, train_svm, sw_search
import cv2, time, argparse, sys
#from db.db import push_train_features, retrieve_train_features
from db.mdb import push_train_features, retrieve_train_features, push_new_job, retrieve_old_job, list_last_jobs, push_bootstrap_features, retrieve_bootstrap_features, finish_job
from utils.interface import find_old_job
import numpy as np

# Reads configuration values from config.ini file
cfg = ConfigParser()
cfg.read('config.ini')

method = cfg.get('configs', 'method')
mode = cfg.get('configs', 'mode')

train_indexes = range(cfg.getint('train_indexes', 'start'), cfg.getint('train_indexes', 'end'), cfg.getint('train_indexes', 'step'))
bootstrap_indexes = range(cfg.getint('bootstrap_indexes', 'start'), cfg.getint('bootstrap_indexes', 'end'), cfg.getint('bootstrap_indexes', 'step'))
test_indexes = range(cfg.getint('test_indexes', 'start'), cfg.getint('test_indexes', 'end'), cfg.getint('test_indexes', 'step'))

params = {
    'folder'    : cfg.get('params', 'folder'),
    'marginX'   : cfg.getint('params', 'marginX'),
    'marginY'   : cfg.getint('params', 'marginY'),
    'neg_weight': cfg.getint('params', 'neg_weight'),
    'method'    : method
}

c_iteration = cfg.getboolean('svm', 'c_iteration')
def_c_value = cfg.getfloat('svm', 'def_c_value')

#################################################################
# If train mode is selected:
#   - Computes backgroung images from train indexes
#   - Starts a new job
#   - Starts training positive and negative features
#################################################################
if mode == 'train':
    details = raw_input("Please enter details about new job: ")

    BG_img = compute_BG_Image(params['folder'], train_indexes)

    job = push_new_job(method, BG_img, details)
    print job['job_id'] + ' job is created at ' + job['timestamp']
    print 'Feature extraction method used: ' + job['method']

    trf, trl, trfc = train(train_indexes, BG_img, params)
    push_train_features(job['job_id'], trf, trl)

    svm = train_svm(trf, trl,' -s 0 -t 0 -c 100')

    trf, trl = bootstrap(bootstrap_indexes, BG_img, params, trf, trl, trfc, svm)
    push_bootstrap_features(job['job_id'], trf, trl)

    svm2 = train_svm(trf, trl, '-s 0 -t 0 -c 100')

###################################################################
# If bootstrap mode is selected:
#   - Lists last x jobs for user to choose which one to load
#   - Retrieves selected training features of selected job
#   - Start bootstrapping
###################################################################
elif mode == 'bootstrap':
    old_job = find_old_job()

    # Copying old job's data into a new job
    old_job_id = old_job['job_id']
    j = retrieve_old_job(old_job_id)
    BG_img = j['BG_img']

    if method != j['method']:
        print 'Method used in ' + old_job_id + ' was: ' + j['method']
        print j['method'] + ' is going to be used rest of the process.'
    method = j['method']

    trf, trl = retrieve_train_features(old_job_id)
    trfc = len(trl)

    details = 'Fork of ' + old_job_id
    job = push_new_job(method, BG_img, details)

    push_train_features(job['job_id'], trf, trl)

    print job['job_id'] + ' job is created at ' + job['timestamp'] + ' as a fork of ' + old_job_id 
    print 'Feature extraction method used: ' + job['method']

    svm = train_svm(trf, trl)

    trf, trl = bootstrap(bootstrap_indexes, BG_img, params, trf, trl, trfc, svm)
    push_bootstrap_features(job['job_id'], trf, trl)

    svm2 = train_svm(trf, trl)

############################################################################
# If sliding window search is selected:
#   - Lists last x jobs for user to choose which one to load
#   - Retrieves selected training features of selected job (after bootstrap)
#   - Starts sliding window search
############################################################################
else:
    old_job = find_old_job()
    # Copying old job's data into a new job
    old_job_id = old_job['job_id']
    j = retrieve_old_job(old_job_id)
    BG_img = j['BG_img']

    if method != j['method']:
        print 'Method used in ' + old_job_id + ' was: ' + j['method']
        print j['method'] + ' is going to be used rest of the process.'
    method = j['method']

    trf, trl = retrieve_train_features(old_job_id)
    trfc = len(trl)

    details = 'Fork of ' + old_job_id
    job = push_new_job(method, BG_img, details)

    push_train_features(job['job_id'], trf, trl)

    print job['job_id'] + ' job is created at ' + job['timestamp'] + ' as a fork of ' + old_job_id 
    print 'Feature extraction method used: ' + job['method']

    #trf, trl = retrieve_bootstrap_features(old_job_id)
    #push_bootstrap_features(job['job_id'], trf, trl)

    svm2 = train_svm(trf, trl,' -s 0 -t 0 -c 10')

svm_AP, svm_PR, svm_RC = sw_search(test_indexes, BG_img, params, svm2)
finish_job(job['job_id'])

print "\nVideo analyze is done! Here are the results: \n"


for ap, index in zip(svm_AP, range(len(svm_AP))):
    print str(index) + "- " + str(ap)

ap_mean = np.mean(svm_AP)
print ap_mean


