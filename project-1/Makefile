CFLAGS = -O2 -std=c++17
#CFLAGS = -O0 -g
LFLAGS =

BINARIES = heat2d

CC=icc

.SECONDARY: 
.PHONY: all 
all: $(BINARIES)

heat2d: auxiliar/auxiliar.o heat2d.o 
	$(CC) $(CFLAGS) $(LFLAGS) -o $@ $^

auxiliar/auxiliar.o:
	$(MAKE) -C auxiliar

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -qopt-report=3 -qopt-report-phase=vec

.PHONY: clean
clean:
	$(MAKE) -C auxiliar clean
	$(RM) $(BINARIES) *.o *.ti *.optrpt 


	
