from mpi4py import MPI
from numpy import array
from PIL import Image
import numpy as np
import colorsys
import sys, getopt

#x, y = -1.4, 0
#w, h = 2048, 2048
#scale = 0.004
#maxit = 1000

#x, y = -1.3, 0
#w, h = 512,512 
#scale = 0.075
#maxit = 500
#filename = "img1.bmp"

x, y = -1.3, 0
w, h = 512, 512
scale = 0.075
maxit = 500
filename = "img1.png"

def color(i,maxit):
	gray = int(255*i/maxit)
	return (0, gray, 100)

def iterations_at_point(x, y, maxit):
	x0 = x
	y0 = y
	iter =0
	while (x*x+y*y<=4) and (iter<maxit):
		xt=x*x-y*y+x0
		yt=2*x*y + y0
		x=xt
		y=yt
		iter+=1
	return iter

# do not modify from this line
def main(argv, rank):
	global x, y
	global w, h
	global scale
	global maxit
	global filename
	try: 
		opts, args = getopt.getopt(argv,"x:y:s:W:H:m:o:h") 
	except getopt.GetoptError: 
		if (rank==0):
			print ('mandel.py -x xcenter -y ycenter -s scale -W width -H height -m maxit -o filename') 
		sys.exit(2) 
	for opt, arg in opts: 
		if opt == '-h': 
			if (rank==0):
				print ('mandel.py -x xcenter -y ycenter -s scale -W width -H height -m maxit -o filename') 
			sys.exit() 
		elif opt in ("-x"): 
			x = float(arg)
		elif opt in ("-y"): 
			y = float(arg)
		elif opt in ("-s"): 
			scale = float(arg)
		elif opt in ("-W"): 
			w = int(arg)
		elif opt in ("-H"): 
			h = int(arg)
		elif opt in ("-m"): 
			maxit = int(arg)
		elif opt in ("-o"): 
			filename = arg 
	if rank==0: 
            print ("mandel: x=", x, "y=", y, "scale=", scale, "width=", w, "height=", h, "maxit=", maxit, "output=", filename)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

main(sys.argv[1:], rank)

start_time = MPI.Wtime()
# do not modify to this line

xmin = x - scale
xmax = x + scale
ymin = y - scale
ymax = y + scale

C = np.array([0]*h*w, dtype='i')
for j in range(h):
    for i in range(w):
        x = xmin + i * (xmax-xmin)/w
        y = ymin + j * (ymax-ymin)/h
        C[j*w+i] = iterations_at_point(x, y, maxit)

image = Image.new('RGB', (w,h))
pixels = image.load()
for j in range(h):
    for i in range(w):
        pixels[i, j] = color(C[j*w+i], maxit)
image.save(filename)

# do not modify from this line
end_time = MPI.Wtime()
if rank == 0:
	print("Overall elapsed time: " + str(end_time-start_time))
