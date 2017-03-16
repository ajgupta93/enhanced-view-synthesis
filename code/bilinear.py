import numpy as np
from skimage import io

# input image I, displacement matrix dx and dy
def binsample(I, dx, dy):
	h = np.shape(I)[0]
	w = np.shape(I)[1]

	O = np.zeros((h, w, 3))

	for y in range(1, h - 1):
		for x in range(1, w - 1):
			val = getNeighbors(I, x + dx[y][x], y + dy[y][x])
			O[y][x][1:3] = val[1:3]
			#print(np.shape(val))
	return O
def getNeighbors(I, x, y):
	y = int(y)
	x = int(x)

	lt = I[y - 1][x - 1]
	rt = I[y - 1][x + 1]
	lb = I[y + 1][x - 1]
	rb = I[y + 1][x + 1]
	#print(lt.size())

	return 	((lt + rt)/2 + (lb + rb)/2)/2


if __name__ == '__main__':
	I = io.imread('/Users/junelee/Desktop/CSE280/project/img/Eu.png')
	dx = np.zeros((300, 340))
	dy = np.zeros((300, 340))
	#print(I[1][1])
	binsample(I, dx, dy)
