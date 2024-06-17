NVCC = nvcc
SRC_DIR = src
BUILD_DIR = build
UTILS_DIR = $(SRC_DIR)/utils

.DEFAULT_GOAL := all

UTIL_HEADERS = $(wildcard $(UTILS_DIR)/*.cuh)
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%, $(wildcard $(SRC_DIR)/*.cu))

$(BUILD_DIR)/%: $(SRC_DIR)/%.cu $(UTIL_HEADERS) | $(BUILD_DIR)
	$(NVCC) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: all
all: $(OBJS)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
