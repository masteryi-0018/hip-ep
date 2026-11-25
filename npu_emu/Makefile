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
SRCS := $(wildcard */*.hip)
HDRS := $(wildcard */*.h)

.PHONY: all test clean
all: $(LIB)

# -ffp-contract=off: every fp32 step is rounded on its own unless written as
# fmaf. A per-file `#pragma clang fp contract(off)` is not enough: HIP's
# __fmul_rn / __fadd_rn are plain operators in a header compiled with
# contract=fast, and after inlining a product feeding a sum becomes one fma.
$(LIB): $(SRCS) $(HDRS)
	@mkdir -p $(BUILD_DIR)
	$(HIPCC) -std=c++17 -O2 -ffp-contract=off -fPIC -shared --offload-arch=$(ARCH) -I. -o $@ $(SRCS)

test: $(LIB)
	NPU_EMU_LIB=$(abspath $(LIB)) $(PYTHON) -m pytest . -q -p no:cacheprovider $(if $(K),-k "$(K)")

clean:
	rm -rf $(BUILD_DIR)
