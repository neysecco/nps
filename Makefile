.PHONY: test clean help

# Folders where we should run make
SUBDIR = fortran
SUBDIR_CLEAN = fortran \
	       python \
	       test \
	       examples
CONFIG_FILE = config.mk
include ${CONFIG_FILE}

default:
	#Go in each folder and run make
	@for subdir in $(SUBDIR) ; \
		do \
			echo; \
			echo "making $@ in $$subdir"; \
			echo; \
			(cd $$subdir && make) || exit 1; \
		done
	@echo ""
	@echo "You can test the build by doing:"
	@echo "    make test"

clean:
	#Go in each folder and run make clean
	@echo " Making clean ... "

	rm -f $(MAKE_CLEAN_ARGUMENTS)
	@for subdir in $(SUBDIR_CLEAN) ; \
		do \
			echo; \
			echo "making $@ in $$subdir"; \
			echo; \
			(cd $$subdir && make $@) || exit 1; \
		done

	@echo " Clean completed"

test:
	(cd test && python3 test_full.py) || exit1; \

report:
	(cd examples/pde_solve && bash run_report.py) || exit1; \

help:
	@echo "Valid extra targets:"
	@echo "    test: runs test_full.py for the regression test"
	@echo "    clean: removes nn_fortran.so, output.txt, *.pyc, *.pickle, and *~"
