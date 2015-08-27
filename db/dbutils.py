import base64
import numpy as np

def from_npfloat_to_base64(num):
	print type(num)
	a = base64.b64encode(num)
	print a
	return a


def from_base64_to_npfloat(b64):
	r = base64.b64encode(b64)
	q = np.frombuffer(r, dtype=np.float64)
	return q[0]
