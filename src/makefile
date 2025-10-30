CXX = g++
CXXFLAGS = -std=c++17 -O3 -mavx2 -Wall -Wextra -pedantic -flto=auto
LDFLAGS =

SRCS = $(wildcard *.cpp)

OBJS = $(SRCS:.cpp=.o)

TARGET = publius

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

strip:
	strip $(TARGET)
