#Specify main directory to load config file
MAIN_DIR = ..
CONFIG_FILE = $(MAIN_DIR)/config.mk
include ${CONFIG_FILE}

nn_fortran.so : precision.o transFuncs.o baseTypes.o steps.o nn_fortran.o nn_fortran.pyf
	$(F2PY) -l$(FC) -c precision.o transFuncs.o baseTypes.o steps.o nn_fortran.o nn_fortran.pyf
	mv nn_fortran*.so $(MAIN_DIR)/python

nn_fortran.pyf : nn_fortran.f90
	$(F2PY) nn_fortran.f90 -m nn_fortran -h nn_fortran.pyf --overwrite-signature

nn_fortran.o : precision.o transFuncs.o baseTypes.o steps.o nn_fortran.f90
	$(FC) -c $(FFLAGS) nn_fortran.f90

steps.o : precision.o steps.f90
	$(FC) -c $(FFLAGS) steps.f90

baseTypes.o : precision.o baseTypes.f90
	$(FC) -c $(FFLAGS) baseTypes.f90

transFuncs.o : precision.o transFuncs.f90
	$(FC) -c $(FFLAGS) transFuncs.f90

precision.o : precision.f90
	$(FC) -c $(FFLAGS) precision.f90

clean:
	@echo "        Cleaning directory: fortran/ ... "
	rm -f $(MAKE_CLEAN_ARGUMENTS)
