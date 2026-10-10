# Builds with the conda environment's gfortran. -O2 without -march or
# -ffast-math keeps the arithmetic plain IEEE double precision.
ifeq ($(origin FC),default)
FC = gfortran
endif
FFLAGS := -O2 -fopenmp -std=f2008 -Wall

geodesic: geodesic.f90
	$(FC) $(FFLAGS) -o $@ $<

clean:
	rm -f geodesic *.mod

.PHONY: clean
