import os
from os.path import join, getsize


counter = 0
indexes = []
for root, dirs, files in os.walk('/Users/Ekrem/Documents/IHA/new_cap'):
	for i in files:
		if os.path.getsize(root + "/" + i) == 0:
			indexes.append(counter)
			indexes.append(counter+1)
			indexes.append(counter+2)
		counter += 1
		#print i + ": " + str(os.path.getsize(root + "/" + i))

print indexes
counter = 0
for root, dirs, files in os.walk('/Users/Ekrem/Documents/IHA/new_cap'):
	for i in files:
		if counter in indexes:
			print "asd"
			os.remove(root + "/" + i)
		counter += 1

"""
for root, dirs, files in os.walk('/Users/Ekrem/Documents/IHA/new_cap'):
	for i in files:
		if os.path.getsize(root + "/" + i) == 0:
			print i


"""
