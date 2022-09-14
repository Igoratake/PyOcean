from numpy import flipud, append, ceil, floor, isnan, nanmin, nanmax, where
from scipy.fftpack import rfft, irfft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, read_table, to_datetime
from astropy.convolution import Gaussian2DKernel, convolve
from numpy import asarray, ndarray
from glob import glob
from pyproj import Proj, transform
import datetime


def DMS2DecimalDegress(X, Y):

    X = X.upper()

    if 'W' in X:
        X = '-'+X.replace('W','')
    else:
        X = X.replace('E','')

    Y = Y.upper()
    if 'S' in Y:
        Y = '-'+Y.replace('S','')
    else:
        Y = Y.replace('N','')

    D, M, S = X.replace(u'°',' ').replace('"'," ").replace("'"," ").split()
    x = D+M/60.+S/3600.
    D, M, S = Y.replace(u'°',' ').replace('"'," ").replace("'"," ").split()
    y = D+M/60.+S/3600.

    return x, y

def DecimalDegress2DMS(X, Y):

    if isinstance(X, (float, int)):
        X = [X]
        Y = [Y]
    if isinstance(X, ndarray):
        X = [X]
        Y = [Y]
    for x, y in zip(X, Y):
        minx = abs(x-int(x))*60
        miny = abs(y-int(y))*60
        segx = abs(minx-int(minx))*60
        segy = abs(miny-int(miny))*60
        print(u"Longitude: {:0.0f} {:0.0f}'{:0.2f}''".format(int(x), float(minx), float(segx)))
        print(u"Latitude: {:0.0f} {:0.0f}'{:0.2f}''".format(int(y), float(miny), float(segy)))

def ConvertCoordinates(x, y, epsgIni, epsgOut):

    inProj = Proj(init='epsg:' + str(epsgIni))
    outProj = Proj(init='epsg:' + str(epsgOut))
    x2, y2 = transform(inProj, outProj, x, y)

    return x2, y2
