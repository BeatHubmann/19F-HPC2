CXX=upcxx
MPICC=mpiicpc
LDFLAGS  += -L../korali/lib -lkorali -lgsl -lgslcblas 
CXXFLAGS += -I../korali/include -I../korali/include/gsl -O3 -g -std=c++17 -Wall -Wfatal-errors
BINARIES = task1_n4
OBJECTS  = model/heat2d.o conduits/single.o conduits/upcxx.o
MPI = task1_n4_mpi
MPIOBJECTS = model/heat2d.o conduits/single_mod_mpi.o conduits/upcxx.o
MPIFLAGS = -I/cluster/apps/intel/parallel_studio_xe_2018_r1/compilers_and_libraries_2018.1.163/linux/mpi/intel64/include -L/cluster/apps/intel/parallel_studio_xe_2018_r1/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib/release_mt -L/cluster/apps/intel/parallel_studio_xe_2018_r1/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib -Xlinker --enable-new-dtags -Xlinker -rpath -Xlinker /cluster/apps/intel/parallel_studio_xe_2018_r1/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib/release_mt -Xlinker -rpath -Xlinker /cluster/apps/intel/parallel_studio_xe_2018_r1/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib -Xlinker -rpath -Xlinker /opt/intel/mpi-rt/2017.0.0/intel64/lib/release_mt -Xlinker -rpath -Xlinker /opt/intel/mpi-rt/2017.0.0/intel64/lib -lmpifort -lmpi -lmpigi -ldl -lrt -lpthread


.SECONDARY:
.PHONY: all
all: $(BINARIES) $(MPI)

$(BINARIES) : % : %.o $(OBJECTS) 
	$(CXX) -o $@ $^ $(LDFLAGS) 

$(MPI) : % : task1_n4.o $(MPIOBJECTS) 
	$(CXX) -o $@ $^ $(LDFLAGS) $(MPIFLAGS)

model/%.o:
	$(MAKE) -C model all

conduits/%.o:
	$(MAKE) -C conduits all

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $<

.PHONY: clean
clean:
	$(MAKE) -C model clean
	$(MAKE) -C conduits clean
	$(RM) $(BINARIES) $(MPI) *.o *.ti *.optrpt *.txt
