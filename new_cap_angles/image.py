import cv2, getpass, requests, json, hashlib

login_url = 'http://localhost/annotation/api/login'
image_post_url = 'http://localhost/annotation/image/post'

i = 0
sample_size = 10
email = ""
pw = ""

def get_credentials():
	email = raw_input("Your email: ")
	pw = getpass.getpass("Your Password: ")
	return email, pw

def login(email, pw):
	payload = {'email': email, 'password': pw}
	r = requests.post(login_url, data=payload)
	if r.status_code is not requests.codes.ok:
		return False
	else:
		return json.loads(r.text)



print "\n\n>>> Connecting to server...\n\n"
s = ""
while True:
	email, pw = get_credentials()
	s = login(email, pw)
	if s is not False:
		break
	else:
		print "\n\n>>> Incorrect email & password combination...\n\n"


for i in range (0, 349):
	imname = "img{:0>5d}.png".format(i)
	files = {'userfile': open(imname, 'rb')}
	payload = {'auth_key': s["Auth_key"], 'hash': hashlib.md5(open(imname, 'rb').read()).hexdigest()}
	r = requests.post(image_post_url, files=files, data=payload)
	print r.text