# Makefile for UVM allocator and LD_PRELOAD interceptor
#
# Usage:
#   make              # Build all
#   make uvm          # Build PyTorch pluggable allocator
#   make preload      # Build LD_PRELOAD interceptor
#   make clean        # Clean build artifacts
#
# To use the LD_PRELOAD interceptor:
#   LD_PRELOAD=./libcudamalloc_managed.so python your_script.py

NVCC ?= nvcc
CXX ?= g++

# CUDA paths
CUDA_HOME ?= /usr/local/cuda
CUDA_INCLUDE = $(CUDA_HOME)/include
CUDA_LIB = $(CUDA_HOME)/lib64

# Compiler flags
CXXFLAGS = -fPIC -O2 -std=c++17
NVCCFLAGS = -Xcompiler -fPIC -O2 -std=c++17
LDFLAGS = -shared -L$(CUDA_LIB) -lcudart -ldl

# Targets
all: uvm_allocator.so libcudamalloc_managed.so

# PyTorch pluggable allocator (same as before)
uvm: uvm_allocator.so

uvm_allocator.so: uvm_allocator.cpp
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $< -lcudart

# LD_PRELOAD library to intercept all cudaMalloc calls
preload: libcudamalloc_managed.so

libcudamalloc_managed.so: cuda_malloc_managed_preload.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -I$(CUDA_INCLUDE) -o $@ $<

clean:
	rm -f uvm_allocator.so libcudamalloc_managed.so

.PHONY: all uvm preload clean
