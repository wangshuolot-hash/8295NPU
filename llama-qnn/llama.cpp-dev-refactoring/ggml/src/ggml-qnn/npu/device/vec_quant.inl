#pragma once

#include "hexagon_npu.h"

#include <hexagon_types.h>

#include <cstdint>

namespace hexagon::vec::quant {

template <typename _TStruct, size_t _Count, auto _MemberPtr> inline HVX_Vector load_into_vector(const _TStruct * src) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TStruct) * _Count, "_TStruct too large for vector load");

    return *reinterpret_cast<const HVX_UVector *>(&(src->*_MemberPtr));
}

template <typename _TStruct, size_t _Count> inline HVX_Vector load_struct_into_vector(const _TStruct * src) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TStruct) * _Count, "_TStruct too large for vector load");

    return *reinterpret_cast<const HVX_UVector *>(src);
}

template <typename _TBlock> inline HVX_Vector load_block_generic(const _TBlock & src) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TBlock), "wrong block size/padding");
    return load_into_vector<_TBlock, 1, &_TBlock::qs>(&src);
}

template <typename _TBlock> inline HVX_Vector make_scale_load_mask() {
    static_assert(sizeof(_TBlock) < hexagon::kBytesPerVector, "wrong block size/padding");
    static_assert(std::is_same_v<decltype(_TBlock::d), npu_device_fp16_t>,
                  "scale field d must be of type npu_device_fp16_t");

    constexpr const size_t kBytesPerScale  = QUANT_BLOCK_SIZE * sizeof(_TBlock::d);
    const size_t           qs_start_offset = offsetof(_TBlock, d);

    hexagon::HVX_VectorAlias ret;
    size_t                   base_i = qs_start_offset;
    for (size_t ret_idx = 0; ret_idx < hexagon::kBytesPerVector; ++ret_idx) {
        const auto offset = ret_idx % kBytesPerScale;
        const auto i      = base_i + (offset % sizeof(_TBlock::d));
        ret.u8[ret_idx]   = (i & 1) ? (i / 2 + 64) : (i / 2);
        if (offset == kBytesPerScale - 1) {
            base_i += sizeof(_TBlock);
        }
    }

    return ret.v;
}

inline size_t default_qs_shuff_idx(size_t idx) {
    return idx;
}

inline size_t q4_qs_shuff_idx(size_t idx) {
    // TODO: The current mask (kIndexShuffle) is hardcoded for the Q4 quantization block layout, where data is arranged in a specific interleaved pattern.
    //   A more general solution would need to programmatically generate the shuffle mask based on the quantization block's structure.
    constexpr const size_t kIndexShuffle[] = {
        0,   4,   8,   12,  16,  20,  24,  28,  32,  36,  40,  44,  48,  52,  56,  60,  2,   6,   10,  14,  18,  22,
        26,  30,  34,  38,  42,  46,  50,  54,  58,  62,  1,   5,   9,   13,  17,  21,  25,  29,  33,  37,  41,  45,
        49,  53,  57,  61,  3,   7,   11,  15,  19,  23,  27,  31,  35,  39,  43,  47,  51,  55,  59,  63,  127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    };
    return kIndexShuffle[idx];
}

template <typename _TBlock, size_t (*_FuncGetShuffIdx)(size_t) = default_qs_shuff_idx>
inline HVX_Vector make_qs_load_mask() {
    static_assert(sizeof(_TBlock) < hexagon::kBytesPerVector, "wrong block size/padding");

    const size_t qs_start_offset = offsetof(_TBlock, qs);
    const size_t qs_end_offset   = qs_start_offset + sizeof(_TBlock::qs);

    hexagon::HVX_VectorAlias ret;
    size_t                   ret_idx = 0;
    for (size_t i = 0; i < hexagon::kBytesPerVector; ++i) {
        auto offset = i % sizeof(_TBlock);
        if (offset >= qs_start_offset && offset < qs_end_offset) {
            size_t idx  = _FuncGetShuffIdx(ret_idx);
            ret.u8[idx] = ((i & 1) ? (i / 2 + 64) : (i / 2));
            ret_idx++;
        }
    }

    return ret.v;
}

template <typename _TBlock>
inline hexagon::HVX_Vector_x2 load_dual_block_generic(const _TBlock *  srcs,
                                                      const HVX_Vector qs_indices,
                                                      const HVX_Vector scale_indices) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TBlock) * 2, "wrong block size/padding");

    const HVX_Vector blocks = load_struct_into_vector<_TBlock, 2>(srcs);

    HVX_Vector block01 = Q6_Vb_vlut32_VbVbI(qs_indices, blocks, 0);
    HVX_Vector scale01 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks, 0);

    block01 = Q6_Vb_vlut32or_VbVbVbI(block01, qs_indices, blocks, 2);
    scale01 = Q6_Vb_vlut32or_VbVbVbI(scale01, scale_indices, blocks, 2);

    if constexpr (sizeof(_TBlock) * 4 > hexagon::kBytesPerVector) {
        block01 = Q6_Vb_vlut32or_VbVbVbI(block01, qs_indices, blocks, 1);
        block01 = Q6_Vb_vlut32or_VbVbVbI(block01, qs_indices, blocks, 3);

        scale01 = Q6_Vb_vlut32or_VbVbVbI(scale01, scale_indices, blocks, 1);
        scale01 = Q6_Vb_vlut32or_VbVbVbI(scale01, scale_indices, blocks, 3);
    }

    hexagon::HVX_Vector_x2 result;
    result.val[0] = block01;
    result.val[1] = scale01;
    return result;
}

template <typename _TBlock>
inline hexagon::HVX_Vector_x3 load_qual_block_generic(const _TBlock *  srcs,
                                                      const HVX_Vector qs_indices,
                                                      const HVX_Vector scale_indices) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TBlock) * 4, "wrong block size/padding");

    hexagon::HVX_Vector_x3 result;

    const HVX_Vector blocks = load_struct_into_vector<_TBlock, 4>(srcs);

    {
        HVX_Vector block0123 = Q6_Vb_vlut32_VbVbI(qs_indices, blocks, 0);
        block0123            = Q6_Vb_vlut32or_VbVbVbI(block0123, qs_indices, blocks, 1);
        block0123            = Q6_Vb_vlut32or_VbVbVbI(block0123, qs_indices, blocks, 2);
        block0123            = Q6_Vb_vlut32or_VbVbVbI(block0123, qs_indices, blocks, 3);

        result.val[0] = block0123;
    }

    {
        HVX_Vector blocks23 = Q6_V_vror_VR(blocks, sizeof(_TBlock) * 2);

        HVX_Vector scale01 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks, 0);
        scale01            = Q6_Vb_vlut32or_VbVbVbI(scale01, scale_indices, blocks, 2);

        HVX_Vector scale23 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks23, 0);
        scale23            = Q6_Vb_vlut32or_VbVbVbI(scale23, scale_indices, blocks23, 2);

        result.val[1] = scale01;
        result.val[2] = scale23;
    }

    return result;
}

template <typename _TBlock>
inline hexagon::HVX_Vector_x4 load_hexa_block_generic(const _TBlock *  srcs,
                                                      const HVX_Vector qs_indices,
                                                      const HVX_Vector scale_indices) {
    static_assert(hexagon::kBytesPerVector >= sizeof(_TBlock) * 6, "wrong block size/padding");

    const HVX_Vector blocks = load_struct_into_vector<_TBlock, 6>(srcs);

    hexagon::HVX_Vector_x4 result;
    {
        HVX_Vector block012345 = Q6_Vb_vlut32_VbVbI(qs_indices, blocks, 0);
        block012345            = Q6_Vb_vlut32or_VbVbVbI(block012345, qs_indices, blocks, 1);
        block012345            = Q6_Vb_vlut32or_VbVbVbI(block012345, qs_indices, blocks, 2);
        block012345            = Q6_Vb_vlut32or_VbVbVbI(block012345, qs_indices, blocks, 3);

        result.val[0] = block012345;
    }

    {
        HVX_Vector blocks23 = Q6_V_vror_VR(blocks, sizeof(_TBlock) * 2);
        HVX_Vector blocks45 = Q6_V_vror_VR(blocks, sizeof(_TBlock) * 4);

        HVX_Vector scale01 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks, 0);
        scale01            = Q6_Vb_vlut32or_VbVbVbI(scale01, scale_indices, blocks, 2);

        HVX_Vector scale23 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks23, 0);
        scale23            = Q6_Vb_vlut32or_VbVbVbI(scale23, scale_indices, blocks23, 2);

        HVX_Vector scale45 = Q6_Vb_vlut32_VbVbI(scale_indices, blocks45, 0);
        scale45            = Q6_Vb_vlut32or_VbVbVbI(scale45, scale_indices, blocks45, 2);

        result.val[1] = scale01;
        result.val[2] = scale23;
        result.val[3] = scale45;
    }

    return result;
}

inline HVX_Vector dequantize_vec_q40_qf16_2blocks(HVX_Vector qs, HVX_Vector scale01, HVX_Vector table) {
    constexpr const uint32_t kSizeOfQs = sizeof(npu_device_block_q4_0::qs);

    HVX_Vector     q_lo = qs;
    HVX_Vector     q_hi = Q6_Vub_vlsr_VubR(qs, 4);
    HVX_VectorPair qp0  = Q6_W_vshuff_VVR(q_hi, q_lo, kSizeOfQs * (1 + 2));

    q_lo = Q6_V_lo_W(qp0);
    q_lo = Q6_Vb_vshuff_Vb(q_lo);
    qp0  = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);

    return Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(qp0), scale01);
}

inline HVX_VectorPair dequantize_vec_q40_qf32_2blocks(HVX_Vector qs, HVX_Vector scale01, HVX_Vector table) {
    constexpr const uint32_t kSizeOfQs = sizeof(npu_device_block_q4_0::qs);

    HVX_Vector     q_lo = qs;
    HVX_Vector     q_hi = Q6_Vub_vlsr_VubR(qs, 4);
    HVX_VectorPair qp0  = Q6_W_vshuff_VVR(q_hi, q_lo, kSizeOfQs * 4);

    q_lo = Q6_V_lo_W(qp0);
    qp0  = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);

    q_lo = Q6_V_lo_W(qp0);

    return Q6_Wqf32_vmpy_VhfVhf(q_lo, scale01);
}

inline HVX_Vector_x2 dequantize_vec_q40_qf16_4blocks(HVX_Vector qs,
                                                     HVX_Vector scale01,
                                                     HVX_Vector scale23,
                                                     HVX_Vector table) {
    constexpr const uint32_t kSizeOfQs = sizeof(npu_device_block_q4_0::qs);

    HVX_Vector q_lo = qs;
    HVX_Vector q_hi = Q6_Vub_vlsr_VubR(qs, 4);

    HVX_VectorPair qp0 = Q6_W_vshuff_VVR(q_hi, q_lo, kSizeOfQs * (1 + 2 + 4));

    q_lo = Q6_V_lo_W(qp0);
    q_lo = Q6_Vb_vshuff_Vb(q_lo);
    qp0  = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);

    q_lo = Q6_V_lo_W(qp0);
    q_hi = Q6_V_hi_W(qp0);

    q_lo = Q6_Vqf16_vmpy_VhfVhf(q_lo, scale01);
    q_hi = Q6_Vqf16_vmpy_VhfVhf(q_hi, scale23);

    hexagon::HVX_Vector_x2 result;
    result.val[0] = q_lo;
    result.val[1] = q_hi;
    return result;
}

inline HVX_VectorPair_x2 dequantize_vec_q40_qf32_4blocks(HVX_Vector qs,
                                                         HVX_Vector scale01,
                                                         HVX_Vector scale23,
                                                         HVX_Vector table) {
    constexpr const uint32_t kSizeOfQs = sizeof(npu_device_block_q4_0::qs);

    HVX_Vector q_lo = qs;
    HVX_Vector q_hi = Q6_Vub_vlsr_VubR(qs, 4);

    HVX_VectorPair qp0 = Q6_W_vshuff_VVR(q_hi, q_lo, kSizeOfQs * 4);

    q_lo = Q6_V_lo_W(qp0);
    qp0  = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);

    q_lo = Q6_V_lo_W(qp0);
    q_hi = Q6_V_hi_W(qp0);

    hexagon::HVX_VectorPair_x2 result;
    result.val[0] = Q6_Wqf32_vmpy_VhfVhf(q_lo, scale01);
    result.val[1] = Q6_Wqf32_vmpy_VhfVhf(q_hi, scale23);
    return result;
}

inline HVX_VectorPair_x3 dequantize_vec_q40_qf32_6blocks(HVX_Vector qs,
                                                         HVX_Vector scale01,
                                                         HVX_Vector scale23,
                                                         HVX_Vector scale45,
                                                         HVX_Vector table) {
    constexpr const uint32_t kSizeOfQs = sizeof(npu_device_block_q4_0::qs);

    HVX_Vector q_lo = qs;
    HVX_Vector q_hi = Q6_Vub_vlsr_VubR(qs, 4);

    HVX_VectorPair qp0 = Q6_W_vshuff_VVR(q_hi, q_lo, kSizeOfQs * 4);

    q_lo = Q6_V_lo_W(qp0);
    q_hi = Q6_V_hi_W(qp0);

    qp0                = Q6_Wh_vlut16_VbVhR_nomatch(q_lo, table, 0);
    HVX_VectorPair qp1 = Q6_Wh_vlut16_VbVhR_nomatch(q_hi, table, 0);

    q_lo          = Q6_V_lo_W(qp0);
    q_hi          = Q6_V_hi_W(qp0);
    HVX_Vector q2 = Q6_V_lo_W(qp1);

    hexagon::HVX_VectorPair_x3 result;
    result.val[0] = Q6_Wqf32_vmpy_VhfVhf(q_lo, scale01);
    result.val[1] = Q6_Wqf32_vmpy_VhfVhf(q_hi, scale23);
    result.val[2] = Q6_Wqf32_vmpy_VhfVhf(q2, scale45);
    return result;
}

inline HVX_Vector load_dequant_vec_q40_qf32_1block(const npu_device_block_q4_0 * src,
                                                   const HVX_Vector              qs_indices,
                                                   const HVX_Vector              scale_indices,
                                                   const HVX_Vector              table) {
    // TODO: can we have a single-block version of load and dequantize?
    auto qs = load_dual_block_generic(src, qs_indices, scale_indices);
    return Q6_V_lo_W(dequantize_vec_q40_qf32_2blocks(qs.val[0], qs.val[1], table));
}

inline HVX_VectorPair load_dequant_vec_q40_qf32_2blocks(const npu_device_block_q4_0 * src,
                                                        const HVX_Vector              qs_indices,
                                                        const HVX_Vector              scale_indices,
                                                        const HVX_Vector              table) {
    auto qs = load_dual_block_generic(src, qs_indices, scale_indices);
    return dequantize_vec_q40_qf32_2blocks(qs.val[0], qs.val[1], table);
}

inline HVX_VectorPair_x2 load_dequant_vec_q40_qf32_4blocks(const npu_device_block_q4_0 * src,
                                                           const HVX_Vector              qs_indices,
                                                           const HVX_Vector              scale_indices,
                                                           const HVX_Vector              table) {
    auto qs = load_qual_block_generic(src, qs_indices, scale_indices);
    return dequantize_vec_q40_qf32_4blocks(qs.val[0], qs.val[1], qs.val[2], table);
}

inline HVX_VectorPair_x3 load_dequant_vec_q40_qf32_6blocks(const npu_device_block_q4_0 * src,
                                                           const HVX_Vector              qs_indices,
                                                           const HVX_Vector              scale_indices,
                                                           const HVX_Vector              table) {
    auto qs = load_hexa_block_generic(src, qs_indices, scale_indices);
    return dequantize_vec_q40_qf32_6blocks(qs.val[0], qs.val[1], qs.val[2], qs.val[3], table);
}

// [v68 bring-up] q8_0 fused dequant: v68 HVX has no fp16 ALU and no int->fp16
// vector conversion, so the scalar fallback in dequantize_row_q8_0 is a
// performance killer for the lm_head GEMV (151936x896 q8_0 matrix). This path
// keeps everything in the v68-supported fp32 HVX domain instead:
//   int8 v = (biased_byte >> 4) * 16 + (biased_byte & 0xF) - 128
// where biased_byte = qs ^ 0x80 turns the signed int8 into [0..255], so the
// high/low nibbles index two 16-entry fp16 tables (values 0,16,..,240 and
// -128..-113) and the scale product + sum happen through Wqf32_vmpy/Vqf32_vadd
// which are available from v66 onwards.
inline HVX_VectorPair dequantize_vec_q80_qf32_2blocks(HVX_Vector qs, HVX_Vector scale01) {
    const HVX_Vector xor80 = Q6_V_vsplat_R(0x80808080u);
    const HVX_Vector and0f = Q6_V_vsplat_R(0x0f0f0f0fu);

    const HVX_Vector v_biased = Q6_V_vxor_VV(qs, xor80);       // int8 -> [0..255]
    const HVX_Vector hi_idx   = Q6_Vub_vlsr_VubR(v_biased, 4);  // [0..15]
    const HVX_Vector lo_idx   = Q6_V_vand_VV(v_biased, and0f);  // [0..15]

    alignas(hexagon::kBytesPerVector) static const HVX_Vector hi_table = [] {
        hexagon::HVX_VectorAlias table;
        table.v = Q6_V_vzero();
        for (int i = 0; i < 16; ++i) {
            table.f16[i * 2] = i * 16;  // same f16[i*2] slot layout as the q4_0 table
        }
        return table.v;
    }();

    alignas(hexagon::kBytesPerVector) static const HVX_Vector lo_table = [] {
        hexagon::HVX_VectorAlias table;
        table.v = Q6_V_vzero();
        for (int i = 0; i < 16; ++i) {
            table.f16[i * 2] = i - 128;  // fold the -128 bias into the table
        }
        return table.v;
    }();

    const HVX_VectorPair hi16 = Q6_Wh_vlut16_VbVhR_nomatch(hi_idx, hi_table, 0);
    const HVX_VectorPair lo16 = Q6_Wh_vlut16_VbVhR_nomatch(lo_idx, lo_table, 0);

    const HVX_Vector hi_f16 = Q6_V_lo_W(hi16);  // first 64 f16 lanes match the 64 input bytes
    const HVX_Vector lo_f16 = Q6_V_lo_W(lo16);

    // a = d * (hi * 16), b = d * (lo - 128) in the fp32 domain
    const HVX_VectorPair a = Q6_Wqf32_vmpy_VhfVhf(hi_f16, scale01);
    const HVX_VectorPair b = Q6_Wqf32_vmpy_VhfVhf(lo_f16, scale01);

    return Q6_W_vcombine_VV(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(a), Q6_V_lo_W(b)),
                            Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(a), Q6_V_hi_W(b)));
}

}  // namespace hexagon::vec::quant
