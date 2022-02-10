F2PY = f2py

FC = gfortran
# Production
FFLAGS = -O2 -fPIC
# Debugging
#FFLAGS = -g -fPIC
#FFLAGS = -g -fPIC -fbounds-check

MAKE_CLEAN_ARGUMENTS = *~ *.o *.mod *.so *.pyc *.pyf *.pickle *.png *.pdf *.mp4
