<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# npu_emu: bit-exact GPU emulation of AIE4 mllib kernels

`libnpu_emu.so` reproduces, bit for bit, NPU (AIE4) kernels of mllib
(`L2/src/wrapper/*_wrapper.cpp`) on an AMD GPU: the conversion kernels and the
data-movement / selection kernels. It is a standalone HIP library with a C ABI
([include/npu_emu.h](include/npu_emu.h)) so any runtime can call it; it does not
depend on the rest of hip-ep.

Tensors use the NPU's real element types (f32, f16, bf16, fp8 E4M3FN, 8/16/32/64-bit
integers, bfp16 and mx6 blocks). The arithmetic follows the kernel source, not
mllib's python reference: on the nightly regression the python reference differs
from aiesim by 1 ULP in 70 of 315 cases (wrong rounding mode, unfused
multiply-add, float64 instead of fp32 coefficients, an extra bf16 rounding),
because mllib accepts a 1-ULP difference.

## Conversion entry points (one per mllib wrapper)

| Class | Functions | NPU arithmetic |
|---|---|---|
| A. format conversion | `npu_convert_fp32_fp16`, `npu_convert_fp16_fp32`, `npu_convert_fp16_bf16`, `npu_cast_bf2bfp16`, `npu_convert_bf16_to_bfp16`, `npu_cast_bfp16tobf`, `npu_cast_bftomx6`, `npu_cast_mx6tobf` | one rounding; block floating point rounds with the wrapper's mode (`cast_bf2bfp16`: half to even, `convert_bf16_to_bfp16` / `cast_bftomx6`: half away from zero) |
| B. fp8 with a scale | `npu_convert_fp32_fp8`, `npu_convert_fp16_fp8`, `npu_convert_fp8_fp32`, `npu_convert_fp8_fp16` | each wrapper differs: fp32 -> fp8 rounds half away and does not saturate; f16/bf16 -> fp8 clamps to +-448 and rounds half to even |
| C. quantize | `npu_quant`, `npu_cast_bfp16toaq` | `sat(rne(fma(x, 1/s, zp)))` |
| D. dequantize | `npu_dequant`, `npu_cast_aqtobfp16`, `npu_dequant_w4_bfp16` | `fma(x, s, -zp*s)`, output rounded half to even |
| E. requantize | `npu_requantize`, `npu_requantize_channelwise`, `npu_requantize_a64`, `npu_cast_int32_int64` | `sat(rne(fma(x, c1, c0)))` with the fp32 coefficients the NPU receives |
| F. quantize-dequantize | `npu_cast_bfp16toqdqf32` | C then D |

Rules every class relies on, all measured on aiesim outputs:

- A multiply-add is fused (one fp32 rounding), and an integer operand enters it
  exactly, even above 2^24 (`requantize_a64` with 32-bit outputs shows it).
- Float to integer rounds half to even and saturates, even though every
  wrapper selects `saturation_mode::truncate`. 32-bit outputs of `dqq.hpp`
  saturate to int32 first, so a uint32 output tops out at 2^31 - 1.
- A bfp16 block whose mantissas are all zero gets exponent 0.
- fp8 is decoded by an 8x8 identity matrix multiply, and NaN x 0 = NaN, so a
  NaN makes all 8 values of its group NaN (8 consecutive elements from the
  buffer start; in an HWC tensor, 8 consecutive channels of one pixel).

Not covered by any aiesim output, so following mllib's python model or IEEE:
NaN, infinity, overflow and subnormal inputs (except `convert_fp16_fp8`, whose
regression covers them), the reciprocal `1.0f / scale` for non power-of-two
scales, and the zero-block exponent of mx6. `npu_convert_bf16_to_bfp16` and
`npu_dequant_w4_bfp16` take logical layouts; the NPU's tiled and packed layouts
are not emulated.

## Data-movement / selection entry points

One entry per mllib wrapper, declared in three headers that `npu_emu.h`
includes. Tensors are logical (the valid region, densely packed); the NPU's
C64 tiling, sub-volumes and garbage padding are not emulated. Each header
documents the NPU behaviour of every function and marks what no aiesim output
covers as unverified.

| Header | Wrappers | NPU arithmetic |
|---|---|---|
| [npu_emu_movement.h](include/npu_emu_movement.h) | concat, slice, copy, pad, transpose, depth/space, expand, gather, nearest resize, not | element bits copied (NaN payloads, -0, fp8 NaN pass through); `concat_unaligned_bfp16` re-encodes blocks that straddle the inputs (half to even); gather indices are read as uint32 and an out-of-range index leaves its output unwritten |
| [npu_emu_select.h](include/npu_emu_select.h) | abs/neg/clip, 3D-broadcast compare and min/max, arg max/min, reduce max/min, max pool, top-k (bf16 / f16 / fp8 / integer) | bf16 VMAX/VMIN order -0 below +0; fp8 kernels compute in fp32 (zero results are +0) and re-encode; arg max / top-k keep the first index of a tie and wrap indices modulo 2^8 or 2^16 |
| [npu_emu_qdq.h](include/npu_emu_qdq.h) | quantized abs/neg/clip/equal/greater/max/min/where, reduce max/min, max pool, concat, gather, nearest grid sample / resize; reshape / slice / transpose with a fused requantization (`*_requantize_fused`) | `sat(round(fma(x, scale, offset) * 2^shift))` with the coefficient words of `coeff32.txt`; round is half to even except `qdq_resize_nearest` (toward zero) |

## Build and test

```bash
make HIPCC=/path/to/hipcc ARCH=gfx1151        # out/libnpu_emu.so
make test HIPCC=/path/to/hipcc PYTHON=python3  # needs numpy and pytest
```

The tests load the library and the HIP runtime (`libamdhip64.so` on the
loader path) through ctypes:

- `tests/test_golden*.py` compare every entry point with the numpy models in
  `tests/golden*.py`. 16-bit and 8-bit float inputs are exhaustive; the rest
  mixes random values with ties, saturation, special values and subnormals.
- `tests/test_aiesim*.py` run every mllib nightly regression case that kept its
  aiesim output through the matching entry point and compares the bytes with
  `data/ofm32.txt` (only the valid region where the regression grades only
  that). Set `NPU_EMU_AIESIM_ROOT` to a
  `.../Vitis_Libraries/mllib/L2/regression/wrapper` tree, or let it pick the
  newest nightly with outputs under `/wrk/mllib_regr`; it is skipped when none
  is reachable.
  `gather_nd_4x3_adf` and `resize_nearest_4x3` only run in the ml_adf graph
  regression; `NPU_EMU_AIESIM_MLADF_ROOT` points at that tree.
