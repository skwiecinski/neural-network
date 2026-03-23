SFML_PATH ?= C:/SFML-3.0
CUDA_PATH ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x

CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++20 -O3 -Wall -Iinclude -I"$(SFML_PATH)/include"

NVCCFLAGS = -O3 -arch=sm_75 -Xcompiler -fPIC -Iinclude

LDFLAGS = -L"$(SFML_PATH)/lib" -L"$(CUDA_PATH)/lib/x64"

LIBS = -lsfml-graphics -lsfml-window -lsfml-system -lcublas -lcudart

CPP_SOURCES = $(wildcard src/*.cpp)
CU_SOURCES = src/CudaMatrixOps.cu

CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)

TARGET = NeuralNetwork

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(TARGET).exe $(TARGET)

.PHONY: all clean
