#!/usr/bin/python
import sys
import h5py
import numpy
f = h5py.File(sys.argv[1],'r')
for k in f.keys():
	print(f[k])
	l = f[k][:]
	print('min %.2f mean %.2f max %.2f'%(l.min(), l.mean(), l.max()))
f.close()
