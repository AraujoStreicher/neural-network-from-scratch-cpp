# Makefile for the Neural Network Project
CXX = g++


CXXFLAGS = -std=c++17 -g -Wall -I include/ -I lib/eigen-3.4.0/
TARGET = neural_network
SOURCES = main.cpp src/Layer.cpp src/NeuralNetwork.cpp

# Automatically generate object file names (.o) from source file names
OBJECTS = $(SOURCES:.cpp=.o)


.PHONY: all
all: $(TARGET)

# Rule to link all the object files (.o) into the final executable (TARGET)
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Rule to compile each source file (.cpp) into an object file (.o)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@



.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJECTS)
