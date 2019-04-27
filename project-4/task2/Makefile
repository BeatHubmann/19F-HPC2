CXX=upcxx
MPICC=mpiicpc
LDFLAGS  += 
CXXFLAGS += -O3 -g -std=c++17 -Wall -Wfatal-errors
BINARIES = single task2a task2b
MPI = task2a_mpi task2b_mpi
OBJECTS = sampler/sampler.o

.SECONDARY:
.PHONY: all
all: $(BINARIES) $(MPI)

$(BINARIES) : % : %.o $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(MPI) : % : %.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

%_mpi.o: %_mpi.cpp
	$(MPICC) -c $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	$(RM) $(BINARIES) $(MPI) *.o *.ti *.optrpt *.txt
