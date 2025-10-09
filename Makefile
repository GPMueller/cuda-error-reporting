# Makefile for Global CUDA Error Handling Example

# Compiler
NVCC = nvcc

# Flags
NVCCFLAGS = -std=c++14 -arch=sm_60 -Xptxas=-v
DEBUG_FLAGS = -g -G -DDEBUG

# Targets
GLOBAL_TARGET = bin/cuda_global_example
MINIMAL_TARGET = bin/minimal_example
GLOBAL_SOURCE = test/cuda_global_error_handler.cu
MINIMAL_SOURCE = test/minimal_example.cu
HEADER = cuda_global_error_handler.h

# Default target - build all examples
all: $(GLOBAL_TARGET) $(MINIMAL_TARGET)

# Build global example
$(GLOBAL_TARGET): $(GLOBAL_SOURCE) $(HEADER)
	$(NVCC) $(NVCCFLAGS) -o $(GLOBAL_TARGET) $(GLOBAL_SOURCE)

# Build minimal example
$(MINIMAL_TARGET): $(MINIMAL_SOURCE) $(HEADER)
	$(NVCC) $(NVCCFLAGS) -o $(MINIMAL_TARGET) $(MINIMAL_SOURCE)

# Debug builds
debug: $(GLOBAL_SOURCE) $(MINIMAL_SOURCE) $(HEADER)
	$(NVCC) $(NVCCFLAGS) $(DEBUG_FLAGS) -o $(GLOBAL_TARGET)_debug $(GLOBAL_SOURCE)
	$(NVCC) $(NVCCFLAGS) $(DEBUG_FLAGS) -o $(MINIMAL_TARGET)_debug $(MINIMAL_SOURCE)

# Clean
clean:
	rm -f $(GLOBAL_TARGET) $(MINIMAL_TARGET) $(GLOBAL_TARGET)_debug $(MINIMAL_TARGET)_debug

# Run examples
run-global: $(GLOBAL_TARGET)
	./$(GLOBAL_TARGET)

run-minimal: $(MINIMAL_TARGET)
	./$(MINIMAL_TARGET)

run-both: $(GLOBAL_TARGET) $(MINIMAL_TARGET)
	@echo "=== Running Global Example ==="
	./$(GLOBAL_TARGET)
	@echo ""
	@echo "=== Running Minimal Example ==="
	./$(MINIMAL_TARGET)

# Help
help:
	@echo "Available targets:"
	@echo "  all         - Build all examples (default)"
	@echo "  debug       - Build with debug flags"
	@echo "  clean       - Remove built files"
	@echo "  run-global  - Build and run the comprehensive example"
	@echo "  run-minimal - Build and run the minimal example"
	@echo "  run-both    - Build and run both examples"
	@echo "  help        - Show this help message"

.PHONY: all debug clean run-global run-minimal run-both help
