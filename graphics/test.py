from graphics import *
def main():
	win = GraphWin("My Circle", 500, 400)
	c = Circle(Point(50,50), 10)
	c.draw(win)
	win.getMouse() # pause for click in window
	win.close()
main()
