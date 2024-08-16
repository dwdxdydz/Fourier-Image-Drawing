#!/usr/bin/python3
# coding: utf-8

import cv2
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import sympy as sp

from draw import drawing

def read_img():
    """
    Read an image from the file system
    IDFK why I made separate fn for it, maybe to put it on top

    Returns:
        numpy.ndarray: The image read from the file system as a NumPy array
    """
    # img = drawing()
    img = cv2.imread('images/Fourier.jpeg')
    # img = cv2.imread('image.jpg')
    # img = img.astype(np.uint8)

    return img


def c_n(list_x, list_y, n):
    """
    Computes the coefficient c_n for a specific frequency component n

    Args:
        list_x (list): List of x-coordinates
        list_y (list): List of y-coordinates
        n (int): Frequency component

    Returns:
        complex: Coefficient c_n.
    """
    N = len(list_x)
    c = 0

    for i in range(N):
        phi = (2 * np.pi * n * i) / N
        c += (list_x[i] - 1j * list_y[i]) * np.exp(-1j * phi)

    c /= N
    return c

def get_coeffs(list_x, list_y, N):
    """
    Calculate the Fourier coefficients for the given x and y coordinate lists

    Args:
        list_x (list): A list of x-coordinates
        list_y (list): A list of y-coordinates
        N (int): The number of Fourier coefficients to compute

    Returns:
        list of tuple: A list of tuples, each containing a complex Fourier coefficient and its frequency
    """
    coefs = [(c_n(list_x, list_y, 0), 0)]

    for i in range(1, N+1):
        coefs.extend([(c_n(list_x, list_y, j), j) for j in (i, -i)])

    return coefs
 

def get_coordinates(img):
    """
    Get the x and y coordinates of the largest contour in the input image
    Additionally it grayscales the image, applies Guassian blur, and make it binary image

    Args:
        img (numpy.ndarray): The input image

    Returns:
        Two lists, `list_x` and `list_y`, representing the x and y coordinates
        of the largest contour after centering around mean of the lists
    """
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgray, (7, 7), 0)

    (T, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour_idx = np.argmax([len(c) for c in contours])

    list_x, list_y = [], []
    for contour in contours[largest_contour_idx]:
        for x,y in contour:
            list_x.append(x)
            list_y.append(y)

    list_x = list_x - np.mean(list_x)
    list_y = list_y - np.mean(list_y)
    
    return list_x, list_y


def get_circle_coords(center, r, N=50):
    """
    Get the coordinates of the cirle of radius 'r' and center 'center'

    Args:
        center (tuple): the center
        r (float): the radius
        N (int, optional): number of points used to approximate teh circle. Defaults to 50

    Returns:
        x, y: two lists of size N of coordinates laying on the circle
    """
    theta = np.linspace(0, 2 * np.pi, N)
    x, y = center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)
    return x, y

def get_next_pos(c, fr, t, drawing_time = 1):
    """
    Get the rotated vector 'c_n' at time t and at frequency fr.

    Args:
        c (complex): c_n 
        fr (_type_): n
        t (float): time t
        drawing_time (int, optional): total drawing time. Defaults to 1.

    Returns:
        rotated vector
    """
    angle = (fr * 2 * np.pi * t) / drawing_time
    return c * np.exp(1j*angle)

def show_image(img):
    """
    Show an image in a window until the 'Esc' key is pressed (For Debugging mostly)

    Args:
        img (numpy.ndarray): The image to be displayed.

    Returns:
        None
    """
    while True:
        cv2.imshow('Canvas', img)
        key = cv2.waitKey(1)
        if key == 27:  # 'Esc' key
            cv2.destroyAllWindows()
            break

def print_eqn():
    t = sp.symbols('t', real=True)

    fourier_series_real = 0
    fourier_series_imag = 0

    for c, n in coefs:
        cn_real = sp.re(c)
        cn_imag = sp.im(c)
        
        fn = 2 * sp.pi * n * t
        rotation_real = cn_real * sp.cos(fn)
        rotation_imag = cn_imag * sp.sin(fn)
        
        fourier_series_real += rotation_real
        fourier_series_imag += rotation_imag

    fourier_series = fourier_series_real + 1j * fourier_series_imag

    print("Equation for Desmos:")
    print(sp.latex(sp.re(fourier_series)) + " + " + sp.latex(sp.im(fourier_series)) + "i")

def see_animation():
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 10))

    circles = [ax.plot([], [], 'b-')[0] for i in range(-N, N+1)]
    circle_lines = [ax.plot([], [], 'g-')[0] for i in range(-N, N+1)]
    drawing, = ax.plot([], [], 'r-', linewidth=2)

    # ax.set_xlim(-5*N, 5*N)
    # ax.set_ylim(-5*N, 5*N)

    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)

    ax.set_axis_off()
    ax.set_aspect('equal')
    # fig.set_size_inches(25, 25)
    
    draw_x, draw_y = [], []
    
    def animate(i, coefs, time): 

        t = time[i]
        
        coefs = [ (get_next_pos(c, fr, t=t), fr) for c, fr in coefs ]
        center = (0, 0)
        for i, elts in enumerate(coefs) :
            c, _ = elts
            r = np.linalg.norm(c)
            x, y = get_circle_coords(center=center, r=r, N=80)
            circle_lines[i].set_data([center[0], center[0]+np.real(c)], [center[1], center[1]+np.imag(c)])
            circles[i].set_data(x, y) 
            center = (center[0] + np.real(c), center[1] + np.imag(c))
        
        # center points now are points from last circle
        # these points are used as drawing points
        draw_x.append(center[0])
        draw_y.append(center[1])

        drawing.set_data(draw_x, draw_y)
    
    drawing_time = 1
    frames = 300
    time = np.linspace(0, drawing_time, num=frames)    
    anim = animation.FuncAnimation(fig, animate, frames = frames, interval = 5, fargs=(coefs, time)) 

    # anim.save('file.gif', writer='pillow', fps=15)
    # plt.show()
    anim.save('file.gif', fps = 15)
    
img = read_img()
list_x, list_y = get_coordinates(img)

N = 300
coefs = get_coeffs(list_x, list_y, N)

see_animation()
