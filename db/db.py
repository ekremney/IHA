from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import create_tables, TrainFeature
from dbutils import from_npfloat_to_base64, from_base64_to_npfloat
engine = create_engine('mysql+mysqldb://root:7171@localhost/iha')
SM = sessionmaker(bind=engine)
session = SM()

create_tables(engine)

def push_train_features(train_features, label):
	index = 0
	for i in train_features:
		for j in range(len(i)):
			tf = TrainFeature(row=index, column=j, value=from_npfloat_to_base64(i[j]), label=label)
			session.merge(tf)
		session.commit()
		index += 1
	print "Train features added to db"

def retrieve_train_features():
	for row in session.query(TrainFeature).all():
		print row
