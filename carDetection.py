from utils.arrayops import array_in_range
from utils.imageops import img_read, read_motion_image, read_bboxes, add_bbox_margin, img_crop, rand_bbox, overlaps, detect_vehicles, non_max_suppression, compute_detection_AP
from random import sample
from libsvm.svmutil import svm_train, svm_predict
from engine.parts import train, test, bootstrap, compute_BG_Image, train_svm
import cv2, time, argparse, sys
#from db.db import push_train_features, retrieve_train_features
from db.mdb import push_train_features, retrieve_train_features, push_new_job, retrieve_old_job, list_last_jobs, push_bootstrap_features, retrieve_bootstrap_features
from utils.interface import find_old_job

#########################################################
# CONSTANTS
#########################################################
DEFAULT_FEAT_EXT = 's_hog'
DEFAULT_MODE = 'search'

#########################################################
# command line arguments: 
# construct the argument parser and parse the arguments
#########################################################
ap = argparse.ArgumentParser(description='Car detection')
ap.add_argument('-m', '--mode', help='Mode of the program. Use "train", "bootstrap", "search"')
ap.add_argument('-t', '--method', help='Method for feature extraction. Use "s_hog", "a_hog", or "lbt"')
args = vars(ap.parse_args())

possible_feats = ['s_hog', 'a_hog', 'lbt']
possible_mods = ['train', 'bootstrap', 'search']
 
# if the mode argument is None, then we are using default mode
# if specified method is not supported, program halts
# otherwise, we are using specified method
if args['mode'] is None:
    mode = DEFAULT_MODE
elif args['mode'] not in possible_mods:
    print 'Specified mode "' + args['mode'] + '" is not supported. Exiting...'
    sys.exit('')
else:
    mode = args['mode']

# if the method argument is None, then we are using default method
# if specified method is not supported, program halts
# otherwise, use specified method as preferred feature extraction method
if args['method'] is None:
    method = DEFAULT_FEAT_EXT
elif args['method'] not in possible_feats:
    print 'Specified method "' + args['method'] + '" is not supported. Exiting...'
    sys.exit(2)
else:
    method = args['method']


###############################################
# VARIABLES
###############################################
train_indexes = array_in_range(3, 3, 90)
bootstrap_indexes = array_in_range(4, 3, 90)
test_indexes = array_in_range(116, 1, 150)

folder = 'trnVideo'
marginX = 5
marginY = 5

params = {
    'folder'    : 'trnVideo',
    'marginX'   : 5,
    'marginY'   : 5,
    'neg_weight': 1,
    'method'    : method
}


#################################################################
# If train mode is selected:
#   - Computes backgroung images from train indexes
#   - Starts a new job
#   - Starts training positive and negative features
#################################################################
if args['mode'] == 'train':
    details = raw_input("Please enter details about new job: ")

    BG_img = compute_BG_Image(folder, train_indexes)

    job = push_new_job(method, BG_img, details)
    print job['job_id'] + ' job is created at ' + job['timestamp']
    print 'Used feature extraction method: ' + job['method']

    trf, trl, trfc = train(train_indexes, BG_img, params)
    push_train_features(job['job_id'], trf, trl)

    svm = train_svm(trf, trl)

    trf, trl, trfc = bootstrap(bootstrap_indexes, BG_img, params, trf, trl, trfc, svm)
    push_bootstrap_features(job['job_id'], trf, trl)

    svm2 = train_svm(trf, trl)


###################################################################
# If bootstrap mode is selecte:
#   - Lists last x jobs for user to choose which one to load
#   - Retrieves selected training features of selected job
#   - Start bootstrapping
###################################################################
elif args['mode'] == 'bootstrap':
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
    new_job = push_new_job(method, BG_img, details)

    push_train_features(new_job['job_id'], trf, trl)

    print new_job['job_id'] + ' job is created at ' + new_job['timestamp'] + ' as a fork of ' + old_job_id 
    print 'Used feature extraction method: ' + new_job['method']

    svm = train_svm(trf, trl)

    trf, trl = bootstrap(bootstrap_indexes, BG_img, params, trf, trl, trfc, svm)
    push_bootstrap_features(new_job['job_id'], trf, trl)

    svm2 = train_svm(trf, trl)

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
    new_job = push_new_job(method, BG_img, details)

    push_train_features(new_job['job_id'], trf, trl)

    print new_job['job_id'] + ' job is created at ' + new_job['timestamp'] + ' as a fork of ' + old_job_id 
    print 'Used feature extraction method: ' + new_job['method']

    trf, trl = retrieve_bootstrap_features(old_job_id)
    push_bootstrap_features(new_job['job_id'], trf, trl)

    svm2 = train_svm(trf, trl)

#time.sleep(100000000)


print "Perform sliding windows search"

# Sliding windows search
svm_AP = [0] * len(test_indexes)
svm_PR = []
svm_RC = []
k = 0

for i in test_indexes:
    img = img_read(folder, i)
    motion_img = read_motion_image(folder, i, BG_img)
    bboxes = read_bboxes(folder, i)

    detections = detect_vehicles(img, motion_img, svm2, marginY, marginX)
    detections = non_max_suppression(detections, 0.01)
    index = 0
    for j in detections:
        img = cv2.rectangle(img,(j[2],j[0]),(j[3],j[1]),(0,255,0),1)
        cv2.putText(img, str(index), (int(j[2]),int(j[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
        index += 1
    filename = "detection{:0>5d}.png".format(k)
    cv2.imwrite(filename, img)
    
    svm_AP[k], svm_PR, svm_RC = compute_detection_AP(detections, bboxes)
    

    print "svm_AP[" + str(k) + "]:"
    print svm_AP[k]

    k += 1
