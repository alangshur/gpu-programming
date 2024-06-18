NVCC = nvcc
SRC_DIR = src
BUILD_DIR = build
UTILS_DIR = $(SRC_DIR)/utils

.DEFAULT_GOAL := all

HEADERS = $(wildcard $(SRC_DIR)/*.cuh $(UTILS_DIR)/*.cuh)
SRCS = $(wildcard $(SRC_DIR)/*.cu $(UTILS_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(patsubst $(UTILS_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRCS)))

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) | $(BUILD_DIR)
	$(NVCC) -I$(SRC_DIR) -I$(UTILS_DIR) -c $< -o $@

$(BUILD_DIR)/%.o: $(UTILS_DIR)/%.cu $(HEADERS) | $(BUILD_DIR)
	$(NVCC) -I$(SRC_DIR) -I$(UTILS_DIR) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: all
all: $(OBJS)
	$(NVCC) $(OBJS) -o main

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
