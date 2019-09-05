#!/usr/bin/python
import sys
import h5py
f = h5py.File(sys.argv[1],'r')
for k in f.keys():
	print(f[k])
f.close()
