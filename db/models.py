from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TrainFeature(Base):
	__tablename__ = 'trainfeatures'

	row = Column(Integer, primary_key=True, autoincrement=False)
	column = Column(Integer, primary_key=True, autoincrement=False)
	value = Column(String(64))
	label = Column(Integer, primary_key=True, autoincrement=False)
	d_type = Column(String(32))

	def __repr__(self):
		return "<TrainFeature row=%d, column=%d, value=%d, label=%d>" % (self.row, self.column, self.value, self.label)

class TestFeature(Base):
	__tablename__ = 'testfeatures'

	row = Column(Integer, primary_key=True, autoincrement=False)
	column = Column(Integer, primary_key=True, autoincrement=False)
	value = Column(String(64))
	label = Column(Integer, primary_key=True, autoincrement=False)
	d_type = Column(String(32))

	def __repr__(self):
		return "<TestFeature row=%d, column=%d, value=%d, label=%d>" % (self.row, self.column, self.value, self.label)

def create_tables(engine):
	Base.metadata.create_all(engine) 
