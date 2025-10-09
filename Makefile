# Makefile for Global CUDA Error Handling Example

# Compiler
NVCC = nvcc

# Flags
NVCCFLAGS = -std=c++14 -arch=sm_60 -Xptxas=-v
DEBUG_FLAGS = -g -G -DDEBUG

# Targets
GLOBAL_TARGET = bin/cuda_global_example
MINIMAL_TARGET = bin/minimal_example
SOFT_TRAP_TARGET = bin/soft_trap_example
GLOBAL_SOURCE = test/cuda_global_error_handler.cu
MINIMAL_SOURCE = test/minimal_example.cu
SOFT_TRAP_SOURCE = test/soft_trap_example.cu
HEADER = cuda_global_error_handler.h
SOFT_TRAP_HEADER = cuda_soft_trap.h

# Default target - build all examples
all: $(GLOBAL_TARGET) $(MINIMAL_TARGET) $(SOFT_TRAP_TARGET)

# Build global example
$(GLOBAL_TARGET): $(GLOBAL_SOURCE) $(HEADER)
	$(NVCC) $(NVCCFLAGS) -o $(GLOBAL_TARGET) $(GLOBAL_SOURCE)

# Build minimal example
$(MINIMAL_TARGET): $(MINIMAL_SOURCE) $(HEADER)
	$(NVCC) $(NVCCFLAGS) -o $(MINIMAL_TARGET) $(MINIMAL_SOURCE)

# Build soft trap example
$(SOFT_TRAP_TARGET): $(SOFT_TRAP_SOURCE) $(SOFT_TRAP_HEADER)
	$(NVCC) $(NVCCFLAGS) -o $(SOFT_TRAP_TARGET) $(SOFT_TRAP_SOURCE)

# Debug builds
debug: $(GLOBAL_SOURCE) $(MINIMAL_SOURCE) $(SOFT_TRAP_SOURCE) $(HEADER) $(SOFT_TRAP_HEADER)
	$(NVCC) $(NVCCFLAGS) $(DEBUG_FLAGS) -o $(GLOBAL_TARGET)_debug $(GLOBAL_SOURCE)
	$(NVCC) $(NVCCFLAGS) $(DEBUG_FLAGS) -o $(MINIMAL_TARGET)_debug $(MINIMAL_SOURCE)
	$(NVCC) $(NVCCFLAGS) $(DEBUG_FLAGS) -o $(SOFT_TRAP_TARGET)_debug $(SOFT_TRAP_SOURCE)

# Clean
clean:
	rm -f $(GLOBAL_TARGET) $(MINIMAL_TARGET) $(SOFT_TRAP_TARGET) $(GLOBAL_TARGET)_debug $(MINIMAL_TARGET)_debug $(SOFT_TRAP_TARGET)_debug

# Run examples
run-global: $(GLOBAL_TARGET)
	./$(GLOBAL_TARGET)

run-minimal: $(MINIMAL_TARGET)
	./$(MINIMAL_TARGET)

run-soft-trap: $(SOFT_TRAP_TARGET)
	./$(SOFT_TRAP_TARGET)

run-both: $(GLOBAL_TARGET) $(MINIMAL_TARGET)
	@echo "=== Running Global Example ==="
	./$(GLOBAL_TARGET)
	@echo ""
	@echo "=== Running Minimal Example ==="
	./$(MINIMAL_TARGET)

run-all: $(GLOBAL_TARGET) $(MINIMAL_TARGET) $(SOFT_TRAP_TARGET)
	@echo "=== Running Global Example ==="
	./$(GLOBAL_TARGET)
	@echo ""
	@echo "=== Running Minimal Example ==="
	./$(MINIMAL_TARGET)
	@echo ""
	@echo "=== Running Soft Trap Example ==="
	./$(SOFT_TRAP_TARGET)

# Help
help:
	@echo "Available targets:"
	@echo "  all            - Build all examples (default)"
	@echo "  debug          - Build with debug flags"
	@echo "  clean          - Remove built files"
	@echo "  run-global     - Build and run the comprehensive example"
	@echo "  run-minimal    - Build and run the minimal example"
	@echo "  run-soft-trap  - Build and run the soft trap example"
	@echo "  run-both       - Build and run global and minimal examples"
	@echo "  run-all        - Build and run all examples"
	@echo "  help           - Show this help message"

.PHONY: all debug clean run-global run-minimal run-soft-trap run-both run-all help
