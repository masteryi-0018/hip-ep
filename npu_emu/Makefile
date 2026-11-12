# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Builds libnpu_emu.so and runs the python tests.
#   make HIPCC=<path to hipcc> [ARCH=gfx1151]
#   make test [PYTHON=python3] [K=<pytest -k expr>]

HIPCC ?= hipcc
ARCH ?= gfx1151
PYTHON ?= python3
BUILD_DIR ?= out

LIB := $(BUILD_DIR)/libnpu_emu.so
SRCS := $(wildcard src/*.hip)
HDRS := $(wildcard include/*.h src/*.h)

.PHONY: all test clean
all: $(LIB)

$(LIB): $(SRCS) $(HDRS)
	@mkdir -p $(BUILD_DIR)
	$(HIPCC) -std=c++17 -O2 -fPIC -shared --offload-arch=$(ARCH) -Iinclude -Isrc -o $@ $(SRCS)

test: $(LIB)
	NPU_EMU_LIB=$(abspath $(LIB)) $(PYTHON) -m pytest tests -q -p no:cacheprovider $(if $(K),-k "$(K)")

clean:
	rm -rf $(BUILD_DIR)
