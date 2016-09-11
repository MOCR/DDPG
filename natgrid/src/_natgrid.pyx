"""
Pyrex wrapper for NCAR natgrid library for interpolation
of irregularly spaced data to a grid.

copyright (c) 2007 by Jeffrey Whitaker.

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notices appear in all copies and that
both the copyright notices and this permission notice appear in
supporting documentation.
THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
"""

# Pyrex generates nnint.c from this file, so make changes here
# and re-create nnint.c with "pyrexc nnint.pyx".

cdef extern void c_nnseti(char *pnam, int ival)
cdef extern double *c_natgridd(int n, double *x, double *y, double *z, int numxout, int numyout, double *xo, double *yo, int *ier)
cdef extern void c_nnsetr(char *pnam, float fval)

cdef extern from "Python.h":
    int PyObject_AsWriteBuffer(object, void **rbuf, Py_ssize_t *len)
    char *PyString_AsString(object)

def seti(name, value):
    cdef char *pnam
    cdef int ival
    pnam = name; ival = value
    c_nnseti(pnam, ival)

def setr(name, value):
    cdef char *pnam
    cdef float fval
    pnam = name; fval = value
    c_nnsetr(pnam, fval)

def natgridd(x, y, z, xo, yo, zo):
    cdef int npnts, numxout, numyout, ier
    cdef Py_ssize_t buflenx, bufleny, buflenz, buflenxo, buflenyo, buflenzo
    cdef void *xp, *yp, *zp, *xop, *yop, *zop
    cdef double *xd, *yd, *zd, *xod, *yod, *zod, *out
    npnts = len(x)
    numxout = len(xo)
    numyout = len(yo)
    if PyObject_AsWriteBuffer(x, &xp, &buflenx) <> 0:
        raise RuntimeError('error getting buffer for x')
    if PyObject_AsWriteBuffer(y, &yp, &bufleny) <> 0:
        raise RuntimeError('error getting buffer for y')
    if PyObject_AsWriteBuffer(z, &zp, &buflenz) <> 0:
        raise RuntimeError('error getting buffer for z')
    xd = <double *>xp
    yd = <double *>yp
    zd = <double *>zp
    if PyObject_AsWriteBuffer(xo, &xop, &buflenxo) <> 0:
        raise RuntimeError('error getting buffer for x')
    if PyObject_AsWriteBuffer(yo, &yop, &buflenyo) <> 0:
        raise RuntimeError('error getting buffer for y')
    if PyObject_AsWriteBuffer(zo, &zop, &buflenzo) <> 0:
        raise RuntimeError('error getting buffer for z')
    xod = <double *>xop
    yod = <double *>yop
    zod = <double *>zop
    # output overwrites zo.
    out = c_natgridd(npnts, yd, xd, zd, numyout, numxout, yod, xod, &ier)
    for i from 0 <= i < buflenzo/8:
        zod[i] = out[i]
    if ier != 0:
       raise RuntimeError('error in natgridd - ier ='%ier)
