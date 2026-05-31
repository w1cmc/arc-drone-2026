ifeq ($(origin CXX),default)
CXX=g++
endif

CXXFLAGS=-g -Wall $(shell pkg-config --cflags --libs opencv4 tbb)

# Separate compile flags and link libraries
CXXFLAGS=-g -Wall $(shell pkg-config --cflags opencv4 tbb)
LIBS=$(shell pkg-config --libs opencv4 tbb) -lavcodec -lavformat -lavutil -lswscale -lpthread

SRCS := circle.cpp opticalFlow.cpp rtsp.cpp
TARGET := circle

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LIBS)

clean:
	rm -f $(TARGET)

