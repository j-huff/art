import time
import gzip
import pickle
import math
from math import acos,sin,cos,pow,sqrt
import numpy as np
import csv
PI = math.pi

def mod(a,b):
	# return a%b
	return (a%b + b)%b


def spiral_points(N,a,f1,f2,f3,f4):
    
    theta = range(N);
    phi =  range(N);
    h =  range(N);
    phi[0] = 0;
    phi[N-1] = 0;
    for k in range(N):
        h[k] = -1 + (2.0*(k))/(N-1.0);
        theta[k] = acos(h[k]);
        
    count = 0
    for k in  range(1,N):
        if sqrt(1.0 - pow(h[k],2.0)) == 0:
            phi[k] = 0
            count+=1
        else:
            # print(phi[k-1] + (3.6/sqrt(N*1.0))* (1.0/sqrt(1.0 - pow(h[k],2.0))))
            phi[k] = mod(phi[k-1] + (3.6/sqrt(N*1.0))* (1.0/sqrt(1.0 - pow(h[k],2.0))),2.0*PI);
    for i in range(N):
        theta[i] = theta[i] - (PI/2);
        theta[i] /= 1
        phi[i]/= 1
    
    
    x = range(N)
    y = range(N)
    z = range(N)
    r = 1.0
    for i in range(N):
        x[i] = r*cos(phi[i]+(a*f1))*sin(theta[i]+(a*f2))
        # x[i] = r*cos(phi[i])
        y[i] = r*sin(phi[i]+(a*f3))*sin(theta[i])
        z[i] = r*cos(theta[i]+a*f4)
        
    return x,y,z

def spiral_points_sph(N):
	
	theta = range(N);
	phi =  range(N);
	h =  range(N);
	phi[0] = 0;
	phi[N-1] = 0;
	for k in range(N):
		h[k] = -1 + (2.0*(k))/(N-1.0);
		theta[k] = acos(h[k]);
		
	count = 0
	for k in  range(1,N):
		if sqrt(1.0 - pow(h[k],2.0)) == 0:
			phi[k] = 0
			count+=1
		else:
			# print(phi[k-1] + (3.6/sqrt(N*1.0))* (1.0/sqrt(1.0 - pow(h[k],2.0))))
			phi[k] = mod(phi[k-1] + (3.6/sqrt(N*1.0))* (1.0/sqrt(1.0 - pow(h[k],2.0))),2.0*PI);
	for i in range(N):
		theta[i] = theta[i] - (PI/2);
		theta[i] /= 1
		phi[i]/= 1
		
	
	for i in range(N):
		phi[i] += PI/2
		theta[i] += PI/2
		phi[i] = mod(phi[i],PI*2)
		theta[i] = mod(theta[i],PI*2)
	
	return phi,theta

def sph2cart(phi,theta):
	N = len(phi)
	x = range(N)
	y = range(N)
	z = range(N)
	r = 1.0

	for i in range(N):
		x[i] = r*cos(phi[i])*sin(theta[i])
		y[i] = r*sin(phi[i])*sin(theta[i])
		# x[i] = 1
		# x[i] = r*cos(phi[i])
	   
		z[i] = r*cos(theta[i])
	return x,y,z
		
def cart2sph(x,y,z):
	N = len(x)
	theta = range(N)
	phi = range(N)
	for i in range(N):
		theta[i]=acos(z[i])
		phi[i]=atan2(y[i],x[i])
		
	# for i in range(N):
	#     x[i] = r*sin(phi[i])*cos(theta[i])
	#     y[i] = r*cos(phi[i])*cos(theta[i])
	#     # x[i] = 1
	#     # x[i] = r*cos(phi[i])
	   
	#     z[i] = r*sin(theta[i])
		
	return phi,theta

def rotate_fun(a,f1,f2,f3,f4):
	def tmp(phi,theta):
		return rotate_point(phi,theta,a,f1,f2,f3,f4)
	return tmp;

def rotate_points(phi,theta,a,f1,f2,f3,f4):
	r = 1.0
	x = r*np.cos(phi+(a*f1))*np.sin(theta+(a*f2))
	y = r*np.sin(phi+(a*f3))*np.sin(theta)
	z = r*np.cos(theta+a*f4)

	return x,y,z

def rotate_point(phi,theta,a,f1,f2,f3,f4):
	r = 1.0
	x = r*cos(phi+(a*f1))*sin(theta+(a*f2))
	y = r*sin(phi+(a*f3))*sin(theta)
	z = r*cos(theta+a*f4)

	return x,y,z

def write_points(x,y,z):
	with open('data_stream', 'wb') as f:
		f.write("points "+repr(len(x))+"\n")
		# for i,_ in enumerate(x):
		# 	# print(repr(x[i])+" "+repr(y[i])+" "+repr(z[i])+"\n")
		# 	f.write(repr(x[i])+" "+repr(y[i])+" "+repr(z[i])+"\n")
		writer = csv.writer(f, delimiter=' ')
		writer.writerows(zip(x,y,z))
		f.write("draw\n")

def clear():
	with open('data_stream', 'wb') as f:
		f.write("clear\n");

phi,theta = spiral_points_sph(1000)
time_steps = 100;
f1 = 1.0
f2 = 2.0
f3 = 3.0
f4 = 4.0
x = []
y = []
z = []


phi = np.array(phi)
theta = np.array(theta)
i = 1
a = i*1.0/time_steps

x,y,z = rotate_points(phi,theta,a,f1,f2,f3,f4) 

# for i in range(time_steps):
# 	print(i)
# 	# x,y,z = spiral_points(1000,i*1.0/time_steps,1.0,2.0,3.0,6.0)
# 	a = i*1.0/time_steps
# 	# fun = rotate_fun(a,f1,f2,f3,f4)
# 	# p = np.array(map(fun,phi,theta))
# 	start = time.time()
# 	x,y,z = rotate_points(phi,theta,a,f1,f2,f3,f4) 
# 	rt = time.time() - start
# 	# print(p)
# 	start = time.time()
# 	clear();
# 	write_points(x,y,z);
# 	wt = time.time() - start

# 	print("Rotate time: "+repr(rt))
# 	print("Write time: "+repr(wt))

# start = time.time()
# phi,theta = spiral_points_sph(10000)
# x,y,z = sph2cart(phi,theta)

# # print(y)

# t = time.time() - start
# print("Time to generate: "+repr(t))

