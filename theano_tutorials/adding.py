# -*- coding: utf-8 -*-

"""
Sample codes for the page('http://deeplearning.net/software/theano/tutorial/adding.html')

"""
__author__ = 'Kaoru Nasuno'

import theano.tensor as T
from theano import function
import numpy as np


def adding_two_scalars():
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y
    f = function([x, y], z)

    print f(2, 3)
    print f(16.3, 12.1)


def adding_two_scalars_shortcut():
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y

    print z.eval({x: 2, y: 3})
    print z.eval({x: 16.3, y: 12.1})  # the second time run is faster because of caches


def adding_two_matrices():
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x + y
    f = function([x, y], z)

    print f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
    print f(np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]]))


def excercise():
    a = T.vector()
    b = T.vector()
    #out = a + a ** 10
    out = a ** 2 + b ** 2 + 2 * a * b
    f = function([a, b], out)
    #print f([0, 1, 2])
    print f([0, 1, 2], [3, 4, 5])


def main():
    #adding_two_scalars()
    #adding_two_scalars_shortcut()
    #adding_two_matrices()
    excercise()


if __name__ == '__main__':
    main()
