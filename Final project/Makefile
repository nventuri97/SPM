# 
# FF_ROOT     points to the FastFlow root directory (i.e.
#             the one containing the ff directory).
ifndef FF_ROOT 
FF_ROOT			= ${HOME}/fastflow
endif

CXX             = g++ -std=c++20
MPI    			= mpicxx -std=c++20
OPTFLAGS	    = -O3 -DNDEBUG -ffast-math
CXXFLAGS       += -Wall

INCLUDES	   	= -I. -I ../include -I $(FF_ROOT)
LIBS            = -pthread
SOURCES         = $(wildcard *.cpp)
TARGET          = $(SOURCES:.cpp=)

.PHONY: all clean cleanall 

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@.o $< $(LIBS)

mpi: MPIUTWavefront.cpp
	$(MPI) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o MPIUTWavefront.o $< $(LIBS)

all: $(TARGET)

# Run the program and calculate the average execution time over 10 runs
run_average:
	@rm -f temp_elapsed_times.txt; \
	for i in $$(seq 1 10); do \
		./$(EXECUTABLE) $(ARGS) | grep "# elapsed time" | awk '{print $$4 " " $$5}' >> temp_elapsed_times.txt; \
	done; \
	./$(EXECUTABLE) $(ARGS) | grep "stimated" >> temp_elapsed_times.txt; \
	echo "$$(grep "stimated" temp_elapsed_times.txt)"; \
	echo "Average elapsed times over 10 executions:"; \
	echo "Sequential code: $$(grep "sequential" temp_elapsed_times.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
	echo "Parallel element cyclic code: $$(grep "element" temp_elapsed_times.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
	echo "Parallel dynamic code: $$(grep "dynamic" temp_elapsed_times.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
	rm -f temp_elapsed_times.txt; 

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)
