import requests
from PIL import Image
from StringIO import StringIO

# First, get metadata of annotated images
r = requests.get('http://localhost/annotation/tool/retrieve_annotated_images')
response = r.json()
count = response['count']
images = response['images']

# Then iterate in metadata; save image to a .png file, save annotation data to .annot file
for i in images:
	image_name = 'img' + i['id'][-5:] + '.png'
	image_path = 'http://localhost' + i['path']
	data_path = 'http://localhost/annotation/tool/retrieve_annotation_data/' + i['id']

	# Fetch annotation for the image and write it to <image.id>.annot file
	r_data = requests.get(data_path)
	annotation = r_data.json()
	
	with open('img' + i['id'][-5:] + '.annot', 'w') as annot_file:
		for i in annotation['annot_data']:
			annot_file.write(i['y1']+","+i['y2']+","+i['x1']+","+i['x2']+"\n")

	# Fetch the image and save it to <image.id>.png file
	r_image = requests.get(image_path, stream=True)
	image = Image.open(StringIO(r_image.content)).convert('LA')
	image.save(image_name)
