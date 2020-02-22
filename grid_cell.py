# Grid Cell Oscillatory Interference Model v0.1
# Code: Zachary Hutchinson (zax) Feb 2020
# Email: zachary.s.hutchinson@maine.edu
#  
# Based on:
# The following grid cell model is based on the versions of an equation found in:
#   - Acker et al 2019 "Stable memory and computation in randomly rewiring neural networks"
# and
#   - Blair el at 2007: "Scale-invariant memory representations emerge from moiré interference 
#   between grid fields that produce theta oscillations: a computational model"
#
# The constants employed are also from Acker2019 (which I believe are also Blair's).
#
# NOTE:
#   For the Acker equation: initial tests indicated that the distance between grid cell 
#   peaks was too small for a given value of lambda (in cm). Referring back to the Blair 
#   paper suggested a correction.
#
#   The Acker paper has: (4pi / sqrt(3l))
#   The Blair paper suggests: (4pi / sqrt(3)) / l
#
#   The Blair version seems to produce grid cell patterns with the correct lambda.
#
#   GridCell_Corrected is based on the Blair paper
#   GridCell_Acker is based on the Acker paper

import math
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
N30_RADIANS = math.radians(-30.0)
P30_RADIANS = math.radians(30.0)
P90_RADIANS = math.radians(90.0)
A = 0.3
B = -1.5
LENGTH = (4.0 * math.pi) / math.sqrt(3.0)

# Corrected ( I think ) based on Blair et al 2007.
def GridCell_Corrected(r, l, theta, c):
    L = LENGTH / l
    R = np.subtract(r,c)
    k1 = math.cos(
        L *
        np.dot(
            [math.cos(N30_RADIANS+theta),math.sin(N30_RADIANS+theta)],
            R
        )
    )
    k2 = math.cos(
        L *
        np.dot(
            [math.cos(P30_RADIANS+theta),math.sin(P30_RADIANS+theta)],
            R
        )
    )
    k3 = math.cos(
        L *
        np.dot(
            [math.cos(P90_RADIANS+theta),math.sin(P90_RADIANS+theta)],
            R
        )
    )

    ksum = k1+k2+k3

    return math.exp( A * (ksum-B) ) - 1.0


# The equation from Acker et al 2019
def GridCell_Acker(r, l, theta, c):
    R = np.subtract(r,c)
    k1 = math.cos(
        (4.0 * math.pi) / math.sqrt(3.0 * l) *
        np.dot(
            [math.cos(N30_RADIANS+theta),math.sin(N30_RADIANS+theta)],
            R
        )
    )
    k2 = math.cos(
        (4.0 * math.pi) / math.sqrt(3.0 * l) *
        np.dot(
            [math.cos(P30_RADIANS+theta),math.sin(P30_RADIANS+theta)],
            R
        )
    )
    k3 = math.cos(
        (4.0 * math.pi) / math.sqrt(3.0 * l) *
        np.dot(
            [math.cos(P90_RADIANS+theta),math.sin(P90_RADIANS+theta)],
            R
        )
    )

    ksum = k1+k2+k3

    return math.exp( A * (ksum-B) ) - 1.0


if __name__ == "__main__":

    # Creates a sqaure environment 100cm by 100cm with 0.1 cm granularity
    field = np.zeros([1000,1000])

    # Distance between grid cell peaks in cm.
    _lambda = 20.0
    # Grid cell orientation.
    _theta = math.radians(30.0)
    # Grid cell origin offset. In cm
    _offset = [0.0,0.0]

    for x in range(1000):
        for y in range(1000):
            field[x][y] = GridCell_Corrected([0.1*x,0.1*y],_lambda,_theta,_offset)

    plt.imshow(field,interpolation='nearest',aspect='auto',cmap='binary')
    plt.show()