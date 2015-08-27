from pymongo import MongoClient
from bson import Binary, Code
import datetime, pickle, bson
from bson.objectid import ObjectId

client = MongoClient('mongodb://localhost:27017/')

db = client.test_database

def push_new_job(method, BG_img, details):
	timestamp = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
	insert = db.jobs.insert({
		'method': method,
		'BG_img': bson.binary.Binary(pickle.dumps(BG_img, protocol=2)),
		'details': details,
		'timestamp': timestamp
		})
	result = {
		'job_id': str(insert),
		'method': method,
		'details': details,
		'timestamp': timestamp
		}
	return result

def retrieve_old_job(job_id):
	job = db.jobs.find_one({"_id": ObjectId(job_id)})
	job['BG_img'] = pickle.loads(job['BG_img'])
	return job

def list_last_jobs(count=5):
	jobs = db.jobs.find().sort([('_id',-1)])
	index = 1
	result = []
	for job in jobs:
		result.append({
			'job_id': str(job['_id']),
			'details': job['details'],
			'timestamp': job['timestamp'],
			'method': job['method']
		})
		print 'No: ' + str(index) + '-'
		print '\tID: ' + str(job['_id'])
		print '\tCreated at: ' + job['timestamp']
		print '\tUsed feature extraction method: ' + job['method']
		print '\tDetails: ' + job['details']
		print '\n------------------------------------------------------\n'
		index += 1
	return result

def push_train_features(job_id, features, labels):
	if len(features) != len(labels):
		return False
	documents = []
	for i in range(len(features)):
		documents.append({
			'row': i,
			'job_id': job_id,
			'feat_array': bson.binary.Binary(pickle.dumps(features[i], protocol=2)),
			'label': labels[i]
			})
	result = db.trainfeatures.insert(documents);
	if len(result) == len(features):
		return True
	else:
		return False

def retrieve_train_features(job_id):
	a = db.trainfeatures.find({"job_id": job_id})
	result = []
	for i in a:
		result.append([i['row'], pickle.loads(i['feat_array']), i['label']])
	result.sort(key=lambda x: x[0])
	trf, trl = [], []
	for i in result:
		trf.append(i[1])
		trl.append(i[2])
	return trf, trl

def push_bootstrap_features(job_id, features, labels):
	if len(features) != len(labels):
		return False
	documents = []
	for i in range(len(features)):
		documents.append({
			'row': i,
			'job_id': job_id,
			'feat_array': bson.binary.Binary(pickle.dumps(features[i], protocol=2)),
			'label': labels[i]
			})
	result = db.bootstrapfeatures.insert(documents);
	if len(result) == len(features):
		return True
	else:
		return False

def retrieve_bootstrap_features(job_id):
	a = db.bootstrapfeatures.find({"job_id": job_id})
	result = []
	for i in a:
		result.append([i['row'], pickle.loads(i['feat_array']), i['label']])
	result.sort(key=lambda x: x[0])
	trf, trl = [], []
	for i in result:
		trf.append(i[1])
		trl.append(i[2])
	return trf, trl	
