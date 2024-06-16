NVCC = nvcc
SRC_DIR = src
BUILD_DIR = build

.DEFAULT_GOAL := all

$(BUILD_DIR)/%: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: all
all: $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%, $(wildcard $(SRC_DIR)/*.cu))

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
