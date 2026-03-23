CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++20 -O3 -Wall -Iinclude

NVCCFLAGS = -O3 -arch=sm_75 -Xcompiler -fPIC

LIBS = -lsfml-graphics -lsfml-window -lsfml-system -lcublas -lcudart

CPP_SOURCES = src/Activation.cpp src/DataLoader.cpp src/Globals.cpp \
              src/Layer.cpp src/LossFunction.cpp src/Matrix.cpp \
              src/Metrics.cpp src/NeuralNetwork.cpp src/Trainer.cpp \
              src/UserManager.cpp src/Utils.cpp src/main.cpp

CU_SOURCES = src/CudaMatrixOps.cu

CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)

TARGET = CudaNeuralNet

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean