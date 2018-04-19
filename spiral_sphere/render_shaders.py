import time
import gzip
import pickle
import math
from math import acos,sin,cos,pow,sqrt
import numpy as np
import csv
import random


from OpenGLContext import testingcontext
BaseContext = testingcontext.getInteractive()

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


# i = 1
# a = i*1.0/time_steps
# a = 1
# x,y,z = rotate_points(phi,theta,a,f1,f2,f3,f4) 
# p = np.array([x,y,z]).T
global SPACE_PRESSED
SPACE_PRESSED = False
def keyboard_fun(e):
	global SPACE_PRESSED
	if e.name == 'space':
		SPACE_PRESSED = True

from OpenGL.arrays import vbo
from OpenGLContext.arrays import *
from OpenGL.GL import shaders


from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import serial
import os
import threading
import colorsys
from PIL import Image
import keyboard
import colour
from colour import Color

keyboard.hook(keyboard_fun)



vs = shaders.compileShader("""#version 330
void main() {
     gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}""", GL_VERTEX_SHADER)
fs = shaders.compileShader("""#version 330
void main() {
    gl_FragColor = vec4( 0, 1, 0, 1 );
}""", GL_FRAGMENT_SHADER)

shader=glCreateProgram()
glAttachShader(shader,vs)
glAttachShader(shader,fs)
glLinkProgram(shader)

# shader = shaders.compileProgram(VERTEX_SHADERp,FRAGMENT_SHADERp)
# log = []
# glGetProgramInfoLog( shader, 1000 , 1000 , log)
# print(log)

x = []
y = []
z = []

height = 1600
width = 800



def get_colors(N):
	c1 = Color(rgb=(1.0, 0.0, 0.0))
	c2 = Color(rgb=(1.0, .5, 0.0))
	c3 = Color(rgb=(1.0, 1.0, 0.0))
	colors1 = list(c1.range_to(c2,N/2))
	colors2 = list(c2.range_to(c3,N/2))

	colors = colors1+colors2
	if len(colors) < N:
		colors.append(c3)
	if len(colors) < N:
		colors.append(c3)
	# print(colors)
	# colors = range(N)
	for i in range(N):
		c = colors[i].rgb
		
		colors[i] = [c[0],c[1],c[2],1.0]
		# h = i*1.0/N
		# c = colorsys.hsv_to_rgb(h/2, 1.0, 1.0)
		# colors[i] = [c[0],c[1],c[2],1.0]
		# # colors[i] = [1.0-h,h,1.0]
	return colors

def get_colors_rainbow(N):
	colors = range(N)
	for i in range(N):
		h = i*1.0/N
		c = colorsys.hsv_to_rgb(h/2, 1.0, 1.0)
		colors[i] = [c[0],c[1],c[2],1.0]
		# colors[i] = [1.0-h,h,1.0]
	return colors


ESCAPE = '\033'
 
window = 0
 
#rotation
X_AXIS = 0.0
Y_AXIS = 0.0
Z_AXIS = 0.0
 
DIRECTION = 1
 
 
def InitGL(Width, Height): 

	glClearColor(0.5, 0.5, 0.5, 0.5)
	glClearDepth(1.0) 
	glDepthFunc(GL_LESS)
	glEnable(GL_DEPTH_TEST)
	glShadeModel(GL_SMOOTH)   
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
	glMatrixMode(GL_MODELVIEW)


def keyPressed(*args):
	if args[0] == ESCAPE:
		sys.exit()


def rotate_points(phi,theta,a,rf,f1,f2,f3,f4,f5):
	r = rf(a)
	a1 = f1(a)
	a2 = f2(a)
	a3 = f3(a)
	a4 = f4(a)
	a5 = f5(a)
	x = r*(np.cos(phi+a1)*np.sin(theta+a2)+np.sin(a)*2)/1.5#+np.cos(theta*a)/10
	y = r*(np.sin(phi+a3)*np.sin(theta+a5)+np.cos(a*.5)*2)/1.5#+np.sin(phi*a)/10
	z = r*(np.cos(theta+a4))#+np.cos(theta+a1)/10

	return x,y,z

N = 100000
phi,theta = spiral_points_sph(N)
# colors = get_colors(N)
colors = get_colors_rainbow(N)

time_steps = 100;
phi = np.array(phi)
theta = np.array(theta)

def linear(a):
	return lambda x : a*x*1.0
global a
a = PI
f1 = linear(1)
f2 = linear(2)
f3 = linear(3)
f4 = linear(4)
f5 = linear(1)
f1 = lambda x : 1
f2 = lambda x : 1
f3 = lambda x : 1
f4 = lambda x : 1
f5 = lambda x : 1
rf = lambda x : .2+x/10

# f1 = lambda x : 1+np.sin(x*5)
# f2 = lambda x : 1+np.cos(x*20)
# f4 = lambda x : 1+np.cos(x*9)
# f5 = lambda x : np.sin(x)

speed = 1.0
point_size = 1.0
intensity = 4.0*speed
rotate_color = False
# tmp = phi
# phi = theta
# theta = tmp
x,y,z = rotate_points(phi,theta,a,rf,f1,f2,f3,f4,f5) 
points = np.array([x,y,z]).T*1.5
# vao_id = glGenVertexArrays(1)
# glBindVertexArray(vao_id)
vbo_id = glGenBuffers(2)
glBindBuffer(GL_ARRAY_BUFFER, vbo_id[0])
glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(points.flatten()), points.flatten(), GL_STATIC_DRAW)
glVertexAttribPointer(shader.attribute_location('vin_position'), 3, GL_FLOAT, GL_FALSE, 0, None)
glEnableVertexAttribArray(0)
glBindBuffer(GL_ARRAY_BUFFER, vbo_id[1])
glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(colors), colors, GL_STATIC_DRAW)
glVertexAttribPointer(shader.attribute_location('vin_color'), 3, GL_FLOAT, GL_FALSE, 0, None)
glEnableVertexAttribArray(1)
glBindVertexArray(0)
# glVertexPointer(3, GL_FLOAT, 0, points.flatten())

global frame_num
frame_num = 0
def DrawGLScene():
	global X_AXIS,Y_AXIS,Z_AXIS
	global DIRECTION
	global a, SPACE_PRESSED, frame_num
	frame_num+=1

	start = time.time()
	a += .001*speed
	x,y,z = rotate_points(phi,theta,a,rf,f1,f2,f3,f4,f5) 
	points = np.array([x,y,z]).T*1.5
	t = time.time()-start;
	print("Move time: "+repr(t))

	glClear(GL_DEPTH_BUFFER_BIT)
	# glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	glLoadIdentity()
	glTranslatef(0.0,0.0,-6.0)

	glRotatef(X_AXIS,1.0,0.0,0.0)
	glRotatef(Y_AXIS,0.0,1.0,0.0)
	glRotatef(Z_AXIS,0.0,0.0,1.0)
	# glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glPointSize(1*point_size);
	# Draw Cube (multiple quads)
	start = time.time()
	
	# pvbo = vbo.VBO(points.flatten())
	# print(np.asarray(points))
	shaders.glUseProgram(shader)
	# pvbo.bind()

	# pvbo.bind()

	glDrawArrays(GL_POINTS, 0, N)
	# pvbo.unbind()
	glDisableClientState(GL_VERTEX_ARRAY);

	t = time.time()-start;
	print("Draw time: "+repr(t))


	# X_AXIS = X_AXIS - 0.01
	# Z_AXIS = Z_AXIS - 0.30
	img = []

	glReadBuffer(GL_BACK);
	if SPACE_PRESSED:
		print("Taking screenshot")
		buffer = glReadPixels(0, 0, width, height, GL_RGB, 
		                         GL_UNSIGNED_BYTE)
		image = Image.frombytes(mode="RGB", size=(width, height), 
		                         data=buffer)
		image = image.transpose(Image.FLIP_TOP_BOTTOM)

		image.save("output-"+repr(time.time())+".png",'PNG')
		SPACE_PRESSED = False

	glutSwapBuffers()

	# time.sleep(.01)


def main():

	global window

	glutInit(sys.argv)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE)
	glutInitWindowSize(width,height)
	glutInitWindowPosition(0,0)

	window = glutCreateWindow('OpenGL Python Cube')

	glutDisplayFunc(DrawGLScene)
	glutIdleFunc(DrawGLScene)
	glutKeyboardFunc(keyPressed)
	InitGL(width, height)
	# glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	glEnable( GL_BLEND);
	glEnable(GL_POINT_SMOOTH)
	glutMainLoop()

if __name__ == "__main__":
	main() 
