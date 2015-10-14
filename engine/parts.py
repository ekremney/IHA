from utils.imageops import median, img_read, read_motion_image, read_bboxes, add_bbox_margin, overlaps, rand_bbox, detect_vehicles, non_max_suppression, compute_detection_AP
from random import sample
from utils.feature_extractor import extract
from tqdm import *
from libsvm.svmutil import svm_train, svm_predict
import numpy as np
import cv2

def compute_BG_Image(folder, train_indexes):
    print "Computing background image"

    s = sample(train_indexes, 7)
    imgs = []
    for i in s:
        imgs.append(img_read(folder, i))
    rows,cols = imgs[0].shape

    bg_img = np.zeros((rows, cols), np.uint8)

    for i in range(rows):
        for j in range(cols):
            mlist = []
            for k in imgs:
                mlist.append(k[i][j])
            bg_img[i,j] = median(mlist)
    return bg_img

def train(train_indexes, BG_img, params):
    folder = params["folder"]
    marginX = params["marginX"]
    marginY = params["marginY"]
    neg_weight = params["neg_weight"]
    method = params["method"]
    feature = params["feature"]

    train_feature_count = 0
    train_features, train_labels = [], []

    print "Extracting positive training features..."
    
    # Read positive train features
    for i in tqdm(range(len(train_indexes))):
        img = img_read(folder, train_indexes[i])
        #motion_img = read_motion_image(folder, train_indexes[i], BG_img)
        height, width = img.shape
        bboxes = add_bbox_margin(read_bboxes(folder, train_indexes[i]), marginX, marginY, height, width)

        for j in bboxes:
            img_cut = img[j[0]:j[1], j[2]:j[3]]
            #motion_img_cut = motion_img[j[0]:j[1], j[2]:j[3]]
            train_feature_count += 1
            #train_features.append(extract(img_cut, motion_img_cut, method, feature))
            train_features.append(extract(img_cut, None, method, feature))
            train_labels.append(1)

    print "Positive training features are extracted."
    print "Extracting negative training features..."

    pos_train_feature_count = train_feature_count
    
    # Read negative train features
    for j in tqdm(range(pos_train_feature_count*neg_weight)):
        i = sample(train_indexes, 1)[0]

        img = img_read(folder, i)

        height, width = img.shape
        bboxes = add_bbox_margin(read_bboxes(folder, i), marginX, marginY, height, width)

        neg_bb = rand_bbox(bboxes, height, width);

        if overlaps(neg_bb, bboxes) != -1:
            continue

        #motion_img = read_motion_image(folder, i, BG_img)

        img_cut = img[neg_bb[0]:neg_bb[1], neg_bb[2]:neg_bb[3]]
        #motion_img_cut = motion_img[neg_bb[0]:neg_bb[1], neg_bb[2]:neg_bb[3]]
        train_feature_count += 1
        #train_features.append(extract(img_cut, motion_img_cut, method, feature))
        train_features.append(extract(img_cut, None, method, feature))
        train_labels.append(-1)
    print "Negative training features are extracted."

    return train_features, train_labels, train_feature_count

def test(test_indexes, BG_img, params):
    folder = params["folder"]
    marginX = params["marginX"]
    marginY = params["marginY"]
    neg_weight = params["neg_weight"]
    method = params["method"]
    feature = params["feature"]

    test_features, test_labels = [], []
    test_feature_count = 0

    print "Extracting positive test features..."

    # Read positive test examples
    for i in tqdm(range(len(test_indexes))):
        img = img_read(folder, test_indexes[i])
        #motion_img = read_motion_image(folder, test_indexes[i], BG_img)

        height, width = img.shape
        bboxes = add_bbox_margin(read_bboxes(folder, test_indexes[i]), marginX, marginY, height, width)

        for j in bboxes:
            img_cut = img[j[0]:j[1], j[2]:j[3]]
            #motion_img_cut = motion_img[j[0]:j[1], j[2]:j[3]]
            test_feature_count += 1
            #test_features.append(extract(img_cut, motion_img_cut, method, feature))
            test_features.append(extract(img_cut, None, method, feature))
            test_labels.append(1)

    pos_test_feature_count = test_feature_count

    print "Positive test features are extracted."
    print "Extracting negative test features..."

    # Read negative test examples
    for j in tqdm(range(pos_test_feature_count*neg_weight)):
        i = sample(test_indexes, 1)[0]

        img = img_read(folder, i)

        height, width = img.shape
        bboxes = add_bbox_margin(read_bboxes(folder, i), marginX, marginY, height, width)

        neg_bb = rand_bbox(bboxes, height, width);

        if overlaps(neg_bb, bboxes) != -1:
            continue

        #motion_img = read_motion_image(folder, i, BG_img)

        img_cut = img[neg_bb[0]:neg_bb[1], neg_bb[2]:neg_bb[3]]
        #motion_img_cut = motion_img[neg_bb[0]:neg_bb[1], neg_bb[2]:neg_bb[3]]
        test_feature_count += 1
        #test_features.append(extract(img_cut, motion_img_cut, method, feature))
        test_features.append(extract(img_cut, None, method, feature))
        test_labels.append(-1)
    
    print "Negative test features are extracted."

    return test_features, test_labels, test_feature_count

def bootstrap(bootstrap_indexes, BG_img, params, trf, trl, trfc, svm):
    folder = params["folder"]
    marginX = params["marginX"]
    marginY = params["marginY"]
    neg_weight = params["neg_weight"]
    method = params["method"]
    feature = params["feature"]

    train_features = trf
    train_labels = trl
    train_feature_count = trfc

    print "Starting bootstrapping..."

    # Bootstrapping
    for i in tqdm(range(len(bootstrap_indexes))):
        img = img_read(folder, bootstrap_indexes[i])
        #motion_img = read_motion_image(folder, bootstrap_indexes[i], BG_img)
        bboxes = read_bboxes(folder, bootstrap_indexes[i])

        #detections = detect_vehicles(img, motion_img, svm, params)
        detections = detect_vehicles(img, None, svm, params)

        hard_negatives = []
        for j in detections:
            if overlaps(j, bboxes) == -1:
                hard_negatives.append(j)

        height, width = img.shape
        hard_negatives = add_bbox_margin(hard_negatives, marginX, marginY, height, width)

        for j in hard_negatives:
            img_cut = img[j[0]:j[1], j[2]:j[3]]
            #motion_img_cut = motion_img[j[0]:j[1], j[2]:j[3]]
            train_feature_count += 1
            #train_features.append(extract(img_cut, motion_img_cut, method, feature))
            train_features.append(extract(img_cut, None, method, feature))
            train_labels.append(-1)
    
    print "Bootstrap finished."

    return train_features, train_labels

def train_svm(train_features, train_labels, arguments='-s 0 -t 0'): # -c param 10-4, 10+4
    print "SVM is being trained"
    svm =  svm_train(train_labels, train_features, arguments)
    print "SVM is trained"
    return svm

def sw_search(test_indexes, BG_img, params, svm):
    print "Performing sliding window search"
    folder = params["folder"]

    svm_AP = [0] * len(test_indexes)
    svm_PR = []
    svm_RC = []
    k = 0

    for i in tqdm(range(len(test_indexes))):
        img = img_read(folder, test_indexes[i])
        #motion_img = read_motion_image(folder, test_indexes[i], BG_img)
        bboxes = read_bboxes(folder, test_indexes[i])

        #detections = detect_vehicles(img, motion_img, svm, params)
        detections = detect_vehicles(img, None, svm, params)
        detections = non_max_suppression(detections, 0.01)

        #index = 0
        for j in detections:
            img = cv2.rectangle(img,(j[2],j[0]),(j[3],j[1]),(0,255,0),1)
            cv2.putText(img, str(j[4])[:5], (int(j[2]),int(j[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255))

        filename = "detection{:0>5d}.png".format(k)
        cv2.imwrite(filename, img)
        
        svm_AP[k], pr, rc = compute_detection_AP(detections, bboxes)

        k += 1

        svm_PR.append(pr)
        svm_RC.append(rc)

    print "Sliding window search is done!"

    return svm_AP, svm_PR, svm_RC

