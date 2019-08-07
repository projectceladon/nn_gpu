/*
 * Copyright @2017 Intel Corporation
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <math.h>
#include <cutils/properties.h>
#include "gpu_executor.h"
#include "vk_common.h"
#include "vk_cs_executor.h"
#include "shader/spv_shader.h"

NAME_SPACE_BEGIN

// TODO: query group count from vulkan device
#define MAX_GROUP_COUNT_X 65535
#define MAX_GROUP_COUNT_Y 65535
#define MAX_GROUP_COUNT_Z 65535

#define SPEC_CONST_NUM 21
#define ITEMS_PER_WI 16

enum ConvShaderType
{
    CONV_SHADER_TYPE_BASIC               = 0,
    CONV_SHADER_TYPE_GEMM1               = 1,
    CONV_SHADER_TYPE_GEMM_4_4_NO_IMG2COL = 2,
    CONV_SHADER_TYPE_GEMM_4_4_GENERIC    = 3,
    CONV_SHADER_TYPE_GEMM_4_4_CHN3       = 4,
    CONV_SHADER_TYPE_GEMM_4_8_GENERIC    = 5,
    CONV_SHADER_TYPE_CHN3_TO_CHN4        = 6,
    CONV_SHADER_TYPE_NUM                 = 7
};


using ShaderConfigPair = std::pair<std::string, std::string>;
using ShaderConfigMap  = std::map<std::string, std::string>;

static std::mutex mtx;
static ShaderConfigMap shaderConfigMap;
static bool is_initialized = false;
static int tmpBoSize = 0;
static int shader_type = CONV_SHADER_TYPE_BASIC;
static bool converted_to_chn4 = false;


struct PushConst {
public:
    PushConst() {};
};

struct SpecializaitonConst {
public:
    SpecializaitonConst() {};
    SpecializaitonConst(int ih, int iw, int oh, int ow, int fh,
                        int fw, int chn, int bat, int M, int K, int N, int tm):
        in_h(ih), in_w(iw), out_h(oh), out_w(ow), filter_h(fh), filter_w(fw),
        channels(chn), batch(bat), m(M), k(K), n(N), tail_m(tm)
    {};

    int local_sz_x;
    int local_sz_y;
    int local_sz_z;
    int in_h;
    int in_w;
    int out_h;
    int out_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int filter_h;
    int filter_w;
    int channels;
    int batch;
    int m;
    int k;
    int n;
    int activation;
    int num_items;    // for chn3tochn4
    int tail_m;       // for gemm_4_4 & gemm_4_8
};

static const char* defaultConfig[] =
{
#ifdef TARGET_GORDON_PEAK
    /* inception-v3 */
    "optype3_batch1_in149_149_32_out147_147_32_filter3_3_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_168_1_block8_4_1",
    "optype3_batch1_in147_147_32_out147_147_64_filter3_3_pad1_1_stride1_1_activation1_bias1", "type5_lsz1_64_1_block8_4_1",
    "optype3_batch1_in73_73_64_out73_73_80_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in73_73_80_out71_71_192_filter3_3_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_72_1_block8_4_1",
    "optype3_batch1_in35_35_192_out35_35_64_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_104_1_block8_4_1",
    "optype3_batch1_in35_35_192_out35_35_48_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_64_1_block8_4_1",
    "optype3_batch1_in35_35_48_out35_35_64_filter5_5_pad2_2_stride1_1_activation1_bias1", "type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in35_35_64_out35_35_96_filter3_3_pad1_1_stride1_1_activation1_bias1", "type5_lsz1_88_1_block8_4_1",
    "optype3_batch1_in35_35_96_out35_35_96_filter3_3_pad1_1_stride1_1_activation1_bias1", "type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in35_35_192_out35_35_32_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_64_1_block8_4_1",
    "optype3_batch1_in35_35_256_out35_35_64_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_5_1_block8_4_1",
    "optype3_batch1_in35_35_256_out35_35_48_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in35_35_288_out35_35_64_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in35_35_288_out35_35_48_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_232_1_block8_4_1",
    "optype3_batch1_in35_35_288_out17_17_384_filter3_3_pad0_0_stride2_2_activation1_bias1", "type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in35_35_96_out17_17_96_filter3_3_pad0_0_stride2_2_activation1_bias1", "type5_lsz1_72_1_block8_4_1",
    "optype3_batch1_in17_17_768_out17_17_192_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in17_17_768_out17_17_128_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_6_1_block8_4_1",
    "optype3_batch1_in17_17_128_out17_17_128_filter1_7_pad0_3_stride1_1_activation1_bias1", "type5_lsz1_184_1_block8_4_1",
    "optype3_batch1_in17_17_128_out17_17_192_filter7_1_pad3_0_stride1_1_activation1_bias1", "type5_lsz1_184_1_block8_4_1",
    "optype3_batch1_in17_17_128_out17_17_128_filter7_1_pad3_0_stride1_1_activation1_bias1", "type5_lsz1_104_1_block8_4_1",
    "optype3_batch1_in17_17_128_out17_17_192_filter1_7_pad0_3_stride1_1_activation1_bias1", "type5_lsz1_72_1_block8_4_1",
    "optype3_batch1_in17_17_768_out17_17_160_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in17_17_160_out17_17_160_filter1_7_pad0_3_stride1_1_activation1_bias1", "type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in17_17_160_out17_17_192_filter7_1_pad3_0_stride1_1_activation1_bias1", "type5_lsz1_72_1_block8_4_1",
    "optype3_batch1_in17_17_160_out17_17_160_filter7_1_pad3_0_stride1_1_activation1_bias1", "type5_lsz1_88_1_block8_4_1",
    "optype3_batch1_in17_17_160_out17_17_192_filter1_7_pad0_3_stride1_1_activation1_bias1", "type5_lsz1_136_1_block8_4_1",
    "optype3_batch1_in17_17_192_out17_17_192_filter1_7_pad0_3_stride1_1_activation1_bias1", "type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in17_17_192_out17_17_192_filter7_1_pad3_0_stride1_1_activation1_bias1", "type5_lsz1_72_1_block8_4_1",
    "optype3_batch1_in17_17_192_out8_8_320_filter3_3_pad0_0_stride2_2_activation1_bias1", "type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in17_17_192_out8_8_192_filter3_3_pad0_0_stride2_2_activation1_bias1", "type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in8_8_1280_out8_8_320_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_152_1_block8_4_1",
    "optype3_batch1_in8_8_1280_out8_8_384_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_64_1_block8_4_1",
    "optype3_batch1_in8_8_384_out8_8_384_filter1_3_pad0_1_stride1_1_activation1_bias1", "type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in8_8_384_out8_8_384_filter3_1_pad1_0_stride1_1_activation1_bias1", "type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in8_8_1280_out8_8_448_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_176_1_block8_4_1",
    "optype3_batch1_in8_8_448_out8_8_384_filter3_3_pad1_1_stride1_1_activation1_bias1", "type5_lsz1_48_1_block8_4_1",
    "optype3_batch1_in8_8_1280_out8_8_192_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_7_1_block8_4_1",
    "optype3_batch1_in8_8_2048_out8_8_320_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_112_1_block8_4_1",
    "optype3_batch1_in8_8_2048_out8_8_384_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_48_1_block8_4_1",
    "optype3_batch1_in8_8_2048_out8_8_448_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_240_1_block8_4_1",
    "optype3_batch1_in8_8_2048_out8_8_192_filter1_1_pad0_0_stride1_1_activation1_bias1", "type5_lsz1_8_1_block8_4_1",
    "optype3_batch1_in1_1_2048_out1_1_1001_filter1_1_pad0_0_stride1_1_activation0_bias1", "type1_lsz4_64_1_block1_1_1",
    /* mobilenet */
    "optype3_batch1_in224_224_3_out112_112_32_filter3_3_pad0_0_stride2_2_activation3_bias1","type4_lsz4_64_1_block4_4_1",
    "optype3_batch1_in112_112_32_out112_112_64_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_64_1_block8_4_1",
    "optype3_batch1_in56_56_64_out56_56_128_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_96_1_block8_4_1",
    "optype3_batch1_in56_56_128_out56_56_128_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in28_28_128_out28_28_256_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_48_1_block8_4_1",
    "optype3_batch1_in28_28_256_out28_28_256_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in14_14_256_out14_14_512_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in14_14_512_out14_14_512_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in7_7_512_out7_7_1024_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_152_1_block8_4_1",
    "optype3_batch1_in7_7_1024_out7_7_1024_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_240_1_block8_4_1",
    "optype3_batch1_in1_1_1024_out1_1_1001_filter1_1_pad0_0_stride1_1_activation0_bias1","type1_lsz4_64_1_block1_1_1",
    /* resnet50 */
    "optype3_batch1_in224_224_3_out112_112_64_filter7_7_pad2_2_stride2_2_activation1_bias1","type4_lsz4_4_1_block4_4_1",
    "optype3_batch1_in56_56_64_out56_56_256_filter1_1_pad0_0_stride1_1_activation0_bias1","type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in56_56_64_out56_56_64_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in56_56_64_out56_56_64_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_104_1_block8_4_1",
    "optype3_batch1_in56_56_256_out56_56_64_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in56_56_256_out28_28_512_filter1_1_pad0_0_stride2_2_activation0_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in56_56_256_out28_28_128_filter1_1_pad0_0_stride2_2_activation1_bias1","type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in28_28_128_out28_28_128_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_32_1_block8_4_1",
    "optype3_batch1_in28_28_128_out28_28_512_filter1_1_pad0_0_stride1_1_activation0_bias1","type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in28_28_512_out28_28_128_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in28_28_512_out14_14_1024_filter1_1_pad0_0_stride2_2_activation0_bias1","type5_lsz1_7_1_block8_4_1",
    "optype3_batch1_in28_28_512_out14_14_256_filter1_1_pad0_0_stride2_2_activation1_bias1","type5_lsz1_7_1_block8_4_1",
    "optype3_batch1_in14_14_256_out14_14_256_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_136_1_block8_4_1",
    "optype3_batch1_in14_14_256_out14_14_1024_filter1_1_pad0_0_stride1_1_activation0_bias1","type5_lsz1_32_1_block8_4_1",
    "optype3_batch1_in14_14_1024_out14_14_256_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_7_1_block8_4_1",
    "optype3_batch1_in14_14_1024_out7_7_2048_filter1_1_pad0_0_stride2_2_activation0_bias1","type5_lsz1_224_1_block8_4_1",
    "optype3_batch1_in14_14_1024_out7_7_512_filter1_1_pad0_0_stride2_2_activation1_bias1","type5_lsz1_168_1_block8_4_1",
    "optype3_batch1_in7_7_512_out7_7_512_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_208_1_block8_4_1",
    "optype3_batch1_in7_7_512_out7_7_2048_filter1_1_pad0_0_stride1_1_activation0_bias1","type5_lsz1_152_1_block8_4_1",
    "optype3_batch1_in7_7_2048_out7_7_512_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_224_1_block8_4_1",
    /* cts */
    "optype3_batch1_in1_1_3_out1_1_3_filter1_1_pad0_0_stride1_1_activation0_bias1", "type0_lsz16_1_1_block1_1_1",
    "optype3_batch1_in2_3_3_out2_3_3_filter1_1_pad0_0_stride1_1_activation0_bias1", "type0_lsz1_1_4_block1_1_1",
    "optype3_batch1_in3_3_1_out2_2_1_filter2_2_pad0_0_stride1_1_activation0_bias1", "type0_lsz1_1_4_block1_1_1",
    "optype3_batch1_in8_8_3_out8_8_1_filter3_2_pad1_0_stride1_1_activation0_bias1", "type0_lsz64_1_4_block1_1_1",
    "optype3_batch1_in8_8_3_out6_7_1_filter3_2_pad0_0_stride1_1_activation0_bias1", "type0_lsz4_4_1_block1_1_1",
    "optype3_batch1_in8_8_3_out8_8_3_filter3_2_pad1_0_stride1_1_activation0_bias1", "type0_lsz256_1_1_block1_1_1",
    "optype3_batch1_in8_8_3_out6_7_3_filter3_2_pad0_0_stride1_1_activation0_bias1", "type0_lsz1_1_1_block1_1_1",
    "optype3_batch1_in224_224_3_out112_112_16_filter3_3_pad0_0_stride2_2_activation3_bias1", "type4_lsz1_16_1_block4_4_1",
    "optype3_batch1_in112_112_16_out112_112_16_filter1_1_pad0_0_stride1_1_activation3_bias1", "type5_lsz1_88_1_block8_4_1",
    "optype3_batch1_in56_56_16_out56_56_32_filter1_1_pad0_0_stride1_1_activation3_bias1", "type5_lsz1_200_1_block8_4_1",
    "optype3_batch1_in56_56_32_out56_56_32_filter1_1_pad0_0_stride1_1_activation3_bias1", "type5_lsz1_120_1_block8_4_1",
    "optype3_batch1_in28_28_32_out28_28_64_filter1_1_pad0_0_stride1_1_activation3_bias1", "type5_lsz1_112_1_block8_4_1",
    "optype3_batch1_in28_28_64_out28_28_64_filter1_1_pad0_0_stride1_1_activation3_bias1", "type5_lsz1_208_1_block8_4_1",
    "optype3_batch1_in14_14_64_out14_14_128_filter1_1_pad0_0_stride1_1_activation3_bias1", "type5_lsz1_152_1_block8_4_1",
    "optype3_batch1_in14_14_128_out14_14_128_filter1_1_pad0_0_stride1_1_activation3_bias1", "type5_lsz1_88_1_block8_4_1",
    "optype3_batch1_in7_7_128_out7_7_256_filter1_1_pad0_0_stride1_1_activation3_bias1", "type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in7_7_256_out7_7_256_filter1_1_pad0_0_stride1_1_activation3_bias1", "type5_lsz1_176_1_block8_4_1",
    "optype3_batch1_in1_1_256_out1_1_11_filter1_1_pad0_0_stride1_1_activation0_bias1", "type1_lsz1_256_1_block1_1_1",
    /* channel 3 to channel 4 transformed*/
    "optype3_batch1_in224_224_4_out112_112_32_filter3_3_pad0_0_stride2_2_activation3_bias1","type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in224_224_4_out112_112_64_filter7_7_pad2_2_stride2_2_activation1_bias1", "type5_lsz1_88_1_block8_4_1",
    "optype3_batch1_in1_1_4_out1_1_3_filter1_1_pad0_0_stride1_1_activation0_bias1", "type1_lsz4_4_1_block1_1_1",
    "optype3_batch1_in2_3_4_out2_3_3_filter1_1_pad0_0_stride1_1_activation0_bias1", "type1_lsz1_4_1_block1_1_1",
    "optype3_batch1_in8_8_4_out8_8_1_filter3_2_pad1_0_stride1_1_activation0_bias1", "type0_lsz4_1_16_block1_1_1",
    "optype3_batch1_in8_8_4_out6_7_1_filter3_2_pad0_0_stride1_1_activation0_bias1", "type0_lsz1_16_16_block1_1_1",
    "optype3_batch1_in8_8_4_out8_8_3_filter3_2_pad1_0_stride1_1_activation0_bias1", "type0_lsz1_4_4_block1_1_1",
    "optype3_batch1_in8_8_4_out6_7_3_filter3_2_pad0_0_stride1_1_activation0_bias1", "type0_lsz1_64_1_block1_1_1",
    "optype3_batch1_in224_224_4_out112_112_16_filter3_3_pad0_0_stride2_2_activation3_bias1", "type5_lsz1_152_1_block8_4_1",
#else
    /* inception-v3 */
    "optype3_batch1_in299_299_4_out149_149_32_filter3_3_pad0_0_stride2_2_activation1_bias1","type5_lsz1_56_1_block8_4_1",
    "optype3_batch1_in149_149_32_out147_147_32_filter3_3_pad0_0_stride1_1_activation1_bias1","type5_lsz1_88_1_block8_4_1",
    "optype3_batch1_in147_147_32_out147_147_64_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_48_1_block8_4_1",
    "optype3_batch1_in73_73_64_out73_73_80_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_224_1_block8_4_1",
    "optype3_batch1_in73_73_80_out71_71_192_filter3_3_pad0_0_stride1_1_activation1_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in35_35_192_out35_35_32_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_208_1_block8_4_1",
    "optype3_batch1_in35_35_192_out35_35_64_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_136_1_block8_4_1",
    "optype3_batch1_in35_35_64_out35_35_96_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in35_35_96_out35_35_96_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_216_1_block8_4_1",
    "optype3_batch1_in35_35_192_out35_35_48_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_208_1_block8_4_1",
    "optype3_batch1_in35_35_48_out35_35_64_filter5_5_pad2_2_stride1_1_activation1_bias1","type5_lsz1_248_1_block8_4_1",
    "optype3_batch1_in35_35_256_out35_35_64_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_176_1_block8_4_1",
    "optype3_batch1_in35_35_256_out35_35_48_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_200_1_block8_4_1",
    "optype3_batch1_in35_35_288_out35_35_64_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_120_1_block8_4_1",
    "optype3_batch1_in35_35_288_out35_35_48_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_224_1_block8_4_1",
    "optype3_batch1_in35_35_96_out17_17_96_filter3_3_pad0_0_stride2_2_activation1_bias1","type5_lsz1_6_1_block8_4_1",
    "optype3_batch1_in35_35_288_out17_17_384_filter3_3_pad0_0_stride2_2_activation1_bias1","type5_lsz1_176_1_block8_4_1",
    "optype3_batch1_in17_17_768_out17_17_192_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in17_17_768_out17_17_128_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_120_1_block8_4_1",
    "optype3_batch1_in17_17_128_out17_17_128_filter7_1_pad3_0_stride1_1_activation1_bias1","type5_lsz1_6_1_block8_4_1",
    "optype3_batch1_in17_17_128_out17_17_128_filter1_7_pad0_3_stride1_1_activation1_bias1","type5_lsz1_120_1_block8_4_1",
    "optype3_batch1_in17_17_128_out17_17_192_filter1_7_pad0_3_stride1_1_activation1_bias1","type5_lsz1_8_1_block8_4_1",
    "optype3_batch1_in17_17_128_out17_17_192_filter7_1_pad3_0_stride1_1_activation1_bias1","type5_lsz1_80_1_block8_4_1",
    "optype3_batch1_in17_17_768_out17_17_160_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in17_17_160_out17_17_160_filter7_1_pad3_0_stride1_1_activation1_bias1","type5_lsz1_32_1_block8_4_1",
    "optype3_batch1_in17_17_160_out17_17_160_filter1_7_pad0_3_stride1_1_activation1_bias1","type5_lsz1_8_1_block8_4_1",
    "optype3_batch1_in17_17_160_out17_17_192_filter1_7_pad0_3_stride1_1_activation1_bias1","type5_lsz1_80_1_block8_4_1",
    "optype3_batch1_in17_17_160_out17_17_192_filter7_1_pad3_0_stride1_1_activation1_bias1","type5_lsz1_8_1_block8_4_1",
    "optype3_batch1_in17_17_192_out17_17_192_filter7_1_pad3_0_stride1_1_activation1_bias1","type5_lsz1_80_1_block8_4_1",
    "optype3_batch1_in17_17_192_out17_17_192_filter1_7_pad0_3_stride1_1_activation1_bias1","type5_lsz1_72_1_block8_4_1",
    "optype3_batch1_in17_17_192_out8_8_192_filter3_3_pad0_0_stride2_2_activation1_bias1","type5_lsz1_5_1_block8_4_1",
    "optype3_batch1_in17_17_192_out8_8_320_filter3_3_pad0_0_stride2_2_activation1_bias1","type5_lsz1_5_1_block8_4_1",
    "optype3_batch1_in8_8_1280_out8_8_192_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_224_1_block8_4_1",
    "optype3_batch1_in8_8_1280_out8_8_448_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in8_8_448_out8_8_384_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in8_8_384_out8_8_384_filter3_1_pad1_0_stride1_1_activation1_bias1","type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in8_8_384_out8_8_384_filter1_3_pad0_1_stride1_1_activation1_bias1","type5_lsz1_8_1_block8_4_1",
    "optype3_batch1_in8_8_1280_out8_8_384_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_56_1_block8_4_1",
    "optype3_batch1_in8_8_1280_out8_8_320_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_7_1_block8_4_1",
    "optype3_batch1_in8_8_2048_out8_8_192_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_160_1_block8_4_1",
    "optype3_batch1_in8_8_2048_out8_8_448_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_152_1_block8_4_1",
    "optype3_batch1_in8_8_2048_out8_8_384_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in8_8_2048_out8_8_320_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in1_1_2048_out1_1_1001_filter1_1_pad0_0_stride1_1_activation0_bias1","type1_lsz1_16_1_block1_1_1",
    /* mobilenet */
    "optype3_batch1_in224_224_3_out112_112_32_filter3_3_pad0_0_stride2_2_activation3_bias1","type4_lsz4_64_1_block4_4_1",
    "optype3_batch1_in112_112_32_out112_112_64_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_80_1_block8_4_1",
    "optype3_batch1_in56_56_64_out56_56_128_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in56_56_128_out56_56_128_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_144_1_block8_4_1",
    "optype3_batch1_in28_28_128_out28_28_256_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_240_1_block8_4_1",
    "optype3_batch1_in28_28_256_out28_28_256_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_96_1_block8_4_1",
    "optype3_batch1_in14_14_256_out14_14_512_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_7_1_block8_4_1",
    "optype3_batch1_in14_14_512_out14_14_512_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_56_1_block8_4_1",
    "optype3_batch1_in7_7_512_out7_7_1024_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_112_1_block8_4_1",
    "optype3_batch1_in7_7_1024_out7_7_1024_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in1_1_1024_out1_1_1001_filter1_1_pad0_0_stride1_1_activation0_bias1","type1_lsz1_16_1_block1_1_1",
    /* resnet50 */
    "optype3_batch1_in224_224_3_out112_112_64_filter7_7_pad2_2_stride2_2_activation1_bias1","type4_lsz4_16_1_block4_4_1",
    "optype3_batch1_in56_56_64_out56_56_256_filter1_1_pad0_0_stride1_1_activation0_bias1","type5_lsz1_64_1_block8_4_1",
    "optype3_batch1_in56_56_64_out56_56_64_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_152_1_block8_4_1",
    "optype3_batch1_in56_56_64_out56_56_64_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_80_1_block8_4_1",
    "optype3_batch1_in56_56_256_out56_56_64_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in56_56_256_out28_28_512_filter1_1_pad0_0_stride2_2_activation0_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in56_56_256_out28_28_128_filter1_1_pad0_0_stride2_2_activation1_bias1","type5_lsz1_256_1_block8_4_1",
    "optype3_batch1_in28_28_128_out28_28_128_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in28_28_128_out28_28_512_filter1_1_pad0_0_stride1_1_activation0_bias1","type5_lsz1_8_1_block8_4_1",
    "optype3_batch1_in28_28_512_out28_28_128_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_88_1_block8_4_1",
    "optype3_batch1_in28_28_512_out14_14_1024_filter1_1_pad0_0_stride2_2_activation0_bias1","type5_lsz1_184_1_block8_4_1",
    "optype3_batch1_in28_28_512_out14_14_256_filter1_1_pad0_0_stride2_2_activation1_bias1","type5_lsz1_80_1_block8_4_1",
    "optype3_batch1_in14_14_256_out14_14_256_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in14_14_256_out14_14_1024_filter1_1_pad0_0_stride1_1_activation0_bias1","type5_lsz1_176_1_block8_4_1",
    "optype3_batch1_in14_14_1024_out14_14_256_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_8_1_block8_4_1",
    "optype3_batch1_in14_14_1024_out7_7_2048_filter1_1_pad0_0_stride2_2_activation0_bias1","type5_lsz1_32_1_block8_4_1",
    "optype3_batch1_in14_14_1024_out7_7_512_filter1_1_pad0_0_stride2_2_activation1_bias1","type5_lsz1_6_1_block8_4_1",
    "optype3_batch1_in7_7_512_out7_7_512_filter3_3_pad1_1_stride1_1_activation1_bias1","type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in7_7_512_out7_7_2048_filter1_1_pad0_0_stride1_1_activation0_bias1","type5_lsz1_216_1_block8_4_1",
    "optype3_batch1_in7_7_2048_out7_7_512_filter1_1_pad0_0_stride1_1_activation1_bias1","type5_lsz1_40_1_block8_4_1",
    /* cts */
    "optype3_batch1_in1_1_3_out1_1_3_filter1_1_pad0_0_stride1_1_activation0_bias1","type0_lsz4_4_4_block1_1_1",
    "optype3_batch1_in2_3_3_out2_3_3_filter1_1_pad0_0_stride1_1_activation0_bias1","type0_lsz1_64_1_block1_1_1",
    "optype3_batch1_in3_3_1_out2_2_1_filter2_2_pad0_0_stride1_1_activation0_bias1","type0_lsz1_16_1_block1_1_1",
    "optype3_batch1_in8_8_3_out8_8_1_filter3_2_pad1_0_stride1_1_activation0_bias1","type0_lsz1_4_16_block1_1_1",
    "optype3_batch1_in8_8_3_out6_7_1_filter3_2_pad0_0_stride1_1_activation0_bias1","type0_lsz4_4_4_block1_1_1",
    "optype3_batch1_in8_8_3_out8_8_3_filter3_2_pad1_0_stride1_1_activation0_bias1","type0_lsz4_16_4_block1_1_1",
    "optype3_batch1_in8_8_3_out6_7_3_filter3_2_pad0_0_stride1_1_activation0_bias1","type0_lsz1_1_16_block1_1_1",
    "optype3_batch1_in224_224_3_out112_112_16_filter3_3_pad0_0_stride2_2_activation3_bias1","type4_lsz4_16_1_block4_4_1",
    "optype3_batch1_in112_112_16_out112_112_16_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_40_1_block8_4_1",
    "optype3_batch1_in56_56_16_out56_56_32_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_168_1_block8_4_1",
    "optype3_batch1_in56_56_32_out56_56_32_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_192_1_block8_4_1",
    "optype3_batch1_in28_28_32_out28_28_64_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_176_1_block8_4_1",
    "optype3_batch1_in28_28_64_out28_28_64_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_16_1_block8_4_1",
    "optype3_batch1_in14_14_64_out14_14_128_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_168_1_block8_4_1",
    "optype3_batch1_in14_14_128_out14_14_128_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_112_1_block8_4_1",
    "optype3_batch1_in7_7_128_out7_7_256_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_3_1_block8_4_1",
    "optype3_batch1_in7_7_256_out7_7_256_filter1_1_pad0_0_stride1_1_activation3_bias1","type5_lsz1_6_1_block8_4_1",
    "optype3_batch1_in1_1_256_out1_1_11_filter1_1_pad0_0_stride1_1_activation0_bias1","type1_lsz4_4_1_block1_1_1",
    /* channel 3 to channel 4 transformed*/
    "optype3_batch1_in1_1_4_out1_1_3_filter1_1_pad0_0_stride1_1_activation0_bias1", "type1_lsz64_1_1_block1_1_1",
    "optype3_batch1_in2_3_4_out2_3_3_filter1_1_pad0_0_stride1_1_activation0_bias1", "type1_lsz1_4_1_block1_1_1",
    "optype3_batch1_in8_8_4_out8_8_1_filter3_2_pad1_0_stride1_1_activation0_bias1", "type0_lsz16_16_1_block1_1_1",
    "optype3_batch1_in8_8_4_out6_7_1_filter3_2_pad0_0_stride1_1_activation0_bias1", "type0_lsz1_4_4_block1_1_1",
    "optype3_batch1_in8_8_4_out8_8_3_filter3_2_pad1_0_stride1_1_activation0_bias1", "type0_lsz4_64_1_block1_1_1",
    "optype3_batch1_in8_8_4_out6_7_3_filter3_2_pad0_0_stride1_1_activation0_bias1", "type0_lsz256_1_1_block1_1_1",
    "optype3_batch1_in224_224_4_out112_112_16_filter3_3_pad0_0_stride2_2_activation3_bias1", "type5_lsz1_248_1_block8_4_1",
    "optype3_batch1_in224_224_4_out112_112_64_filter7_7_pad2_2_stride2_2_activation1_bias1", "type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in224_224_4_out112_112_32_filter3_3_pad0_0_stride2_2_activation3_bias1", "type5_lsz1_56_1_block8_4_1",
    /* temporarily for unit test purpose */
    "optype3_batch1_in3_3_4_out3_3_1_filter1_1_pad0_0_stride1_1_activation0_bias1", "type1_lsz1_4_1_block1_1_1",
    "optype3_batch1_in7_7_1024_out7_7_1024_filter1_1_pad0_0_stride1_1_activation3_bias1", "type5_lsz1_24_1_block8_4_1",
    "optype3_batch1_in8_8_8_out6_6_8_filter3_3_pad0_0_stride1_1_activation0_bias1", "type5_lsz1_1_1_block8_4_1",
    "optype3_batch1_in7_7_8_out5_5_8_filter3_3_pad0_0_stride1_1_activation0_bias1", "type5_lsz1_1_1_block8_4_1",
    "optype3_batch1_in8_8_4_out6_6_8_filter3_3_pad0_0_stride1_1_activation0_bias1", "type5_lsz1_256_1_block8_4_1"
#endif
};

static bool computeGroupCount(int& gx, int& gy, int& gz, const int type,
                              const SpecializaitonConst& param, const ShaderConfig& conf)
{
    switch (type)
    {
    case CONV_SHADER_TYPE_BASIC: {
        gx = alignSize(alignSize(param.n, conf.block_width) / conf.block_width, param.local_sz_x) / param.local_sz_x;
        gy = alignSize(alignSize(param.m, conf.block_height) / conf.block_height, param.local_sz_y) / param.local_sz_y;
        gz = alignSize(alignSize(param.batch, conf.block_depth), param.local_sz_z) / param.local_sz_z;
        break;
    }
    case CONV_SHADER_TYPE_GEMM1: {
        ASSERT(conf.block_width == 1 && conf.block_height == 1 && conf.block_depth == 1 && conf.local_size_z == 1);
        gx = alignSize(param.n, conf.local_size_x) / conf.local_size_x;
        gy = alignSize(param.m, conf.local_size_y) / conf.local_size_y;
        gz = param.batch;
        break;
    }
    case CONV_SHADER_TYPE_GEMM_4_4_NO_IMG2COL: {
        ASSERT(conf.block_width == 4 && conf.block_height == 4 && conf.block_depth == 1 && conf.local_size_z == 1);
        gx = alignSize(param.n / 4, conf.local_size_x) / conf.local_size_x;
        gy = alignSize(alignSize(param.m, 4) / 4, conf.local_size_y) / conf.local_size_y;
        gz = param.batch;
        break;
    }
    case CONV_SHADER_TYPE_GEMM_4_4_GENERIC: {
        ASSERT(conf.block_width == 4 && conf.block_height == 4 && conf.block_depth == 1 && conf.local_size_z == 1);
        gx = alignSize(param.n / 4, conf.local_size_x) / conf.local_size_x;
        gy = alignSize(alignSize(param.m, 4) / 4, conf.local_size_y) / conf.local_size_y;
        gz = param.batch;
        break;
    }
    case CONV_SHADER_TYPE_GEMM_4_8_GENERIC: {
        ASSERT(conf.block_width == 8 && conf.block_height == 4 && conf.block_depth == 1 && conf.local_size_z == 1);
        gx = alignSize(param.n / conf.block_width, conf.local_size_x) / conf.local_size_x;
        gy = alignSize(alignSize(param.m, 4) / 4, conf.local_size_y) / conf.local_size_y;
        gz = param.batch;
        break;
    }
    case CONV_SHADER_TYPE_GEMM_4_4_CHN3: {
        ASSERT(param.m % 4 == 0 && conf.block_width == 4);
        ASSERT(conf.block_height == 4 && conf.block_depth == 1 && conf.local_size_z == 1);
        gx = alignSize(param.n / 4, conf.local_size_x) / conf.local_size_x;
        gy = alignSize(param.m / 4, conf.local_size_y) / conf.local_size_y;
        gz = param.batch;
        break;
    }
    default:
        NOT_REACH_HERE;
        break;
    }

    // todo: validates group size
    return true;
}

static bool chn3ToChn4(SpecializaitonConst& param, ShaderConfig& config)
{
    param.local_sz_x   = 16;
    param.local_sz_y   = 1;
    param.local_sz_z   = 1;

    config.block_width  = ITEMS_PER_WI;
    config.block_height = 1;
    config.block_depth  = 1;

    return true;
}

void computeConvOutputShapeAndPadding(const PaddingScheme& padding_mode,
                                      const uint32_t& in_h, const uint32_t& in_w,
                                      const uint32_t& filter_h, const uint32_t& filter_w,
                                      const uint32_t& dilation_h, const uint32_t& dilation_w,
                                      const uint32_t& stride_h, const uint32_t& stride_w,
                                      uint32_t& out_h, uint32_t& out_w)
{
    if (padding_mode == kPaddingValid)
    {
        out_h = ceil((in_h - (filter_h - 1) * dilation_h) / stride_h);
        out_w = ceil((in_w - (filter_w - 1) * dilation_w) / stride_w);
    }
    else if (padding_mode == kPaddingSame)
    {
        out_h = ceil(in_h / stride_h);
        out_w = ceil(in_w / stride_w);
    }
    else
    {
        LOGE("Invalid padding mode:%d", padding_mode);
    }
}

static void prepareConfig(const Operation& operation, ShaderConfig& config)
{
    //tune();
    (void)(operation);
    (void)(config);
}

static std::string genConvSignature(const SpecializaitonConst& convParam)
{
    std::stringstream sig;
    // Android NN desn't set has_bias, so assume has_bias = 1
    int has_bias = 1;

    sig << "optype"     << (int)OperationType::CONV_2D << "_"
        << "batch"      << convParam.batch             << "_"
        << "in"         << convParam.in_h              << "_" << convParam.in_w     << "_" << convParam.channels << "_"
        << "out"        << convParam.out_h             << "_" << convParam.out_w    << "_" << convParam.n        << "_"
        << "filter"     << convParam.filter_h          << "_" << convParam.filter_w << "_"
        << "pad"        << convParam.pad_h             << "_" << convParam.pad_w    << "_"
        << "stride"     << convParam.stride_h          << "_" << convParam.stride_w << "_"
        << "activation" << convParam.activation        << "_"
        << "bias"       << has_bias;

    return sig.str();
}

static void string2Config(const char* confString, ShaderConfig &conf)
{
    sscanf(confString, "type%d_lsz%d_%d_%d_block%d_%d_%d",
           &shader_type, &conf.local_size_x,  &conf.local_size_y, &conf.local_size_z,
           &conf.block_width, &conf.block_height, &conf.block_depth);

    NN_GPU_DEBUG("CONV_2D: string2Config shader type is %d, local_size_x %d, local_size_y %d, local_size_z %d, "
                 "block_width %d, block_height %d, block_depth %d",
                 shader_type, conf.local_size_x, conf.local_size_y,
                 conf.local_size_z, conf.block_width, conf.block_height, conf.block_depth);
}

static void prepareShaderConfig(const SpecializaitonConst& convParam, ShaderConfig& conf)
{
    const std::string sig = genConvSignature(convParam);

    mtx.lock();

    // load default configs and get vulkan info
    if (!is_initialized)
    {
        NN_GPU_DEBUG("prepareShaderConfig: init shaderConfigMap for vulkan backend shader");
        int configNum = 0;
        if (sizeof(defaultConfig) > 0)
        {
            configNum = sizeof(defaultConfig) / sizeof(defaultConfig[0]) / 2;
        }
        for (int i = 0; i < configNum; i++)
        {
            ShaderConfigPair entry(defaultConfig[2 * i], defaultConfig[2 * i + 1]);
            shaderConfigMap.insert(entry);
            NN_GPU_PERF("CONV_2D: %s: load pre-tuned config: %s, %s\n", __func__, defaultConfig[2 * i], defaultConfig[2 * i + 1]);
        }
        NN_GPU_DEBUG("prepareShaderConfig: shaderConfigMap is initialized");
        is_initialized = true;
    }

    // search in-memory cache
    ShaderConfigMap::iterator it = shaderConfigMap.find(sig);
    if (it != shaderConfigMap.end())
    {
        NN_GPU_PERF("CONV_2D: %s: found config %s, %s\n", __func__, sig.c_str(), it->second.c_str());
        string2Config(it->second.c_str(), conf);
        mtx.unlock();
        return;
    }

    // todo: load from persistent storage & tuning
    NN_GPU_PERF("CONV_2D: %s: config cannot be found from in-memory cache", __func__);

    mtx.unlock();
}


static void setSpecInfo(VkSpecializationMapEntry* entry,
                        VkSpecializationInfo& spec_info,
                        const SpecializaitonConst &spec_const,
                        const int entry_size)
{
    SET_SPEC_CONST_ENTRY(entry[0], 0, offsetof(SpecializaitonConst, local_sz_x), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[1], 1, offsetof(SpecializaitonConst, local_sz_y), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[2], 2, offsetof(SpecializaitonConst, local_sz_z), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[3], 3, offsetof(SpecializaitonConst, in_h), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[4], 4, offsetof(SpecializaitonConst, in_w), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[5], 5, offsetof(SpecializaitonConst, out_h), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[6], 6, offsetof(SpecializaitonConst, out_w), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[7], 7, offsetof(SpecializaitonConst, stride_h), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[8], 8, offsetof(SpecializaitonConst, stride_w), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[9], 9, offsetof(SpecializaitonConst, pad_h), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[10], 10, offsetof(SpecializaitonConst, pad_w), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[11], 11, offsetof(SpecializaitonConst, filter_h), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[12], 12, offsetof(SpecializaitonConst, filter_w), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[13], 13, offsetof(SpecializaitonConst, channels), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[14], 14, offsetof(SpecializaitonConst, batch), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[15], 15, offsetof(SpecializaitonConst, m), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[16], 16, offsetof(SpecializaitonConst, k), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[17], 17, offsetof(SpecializaitonConst, n), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[18], 18, offsetof(SpecializaitonConst, activation), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[19], 19, offsetof(SpecializaitonConst, num_items), sizeof(int));
    SET_SPEC_CONST_ENTRY(entry[20], 20, offsetof(SpecializaitonConst, tail_m), sizeof(int));

    spec_info.mapEntryCount = entry_size;
    spec_info.pMapEntries   = entry;
    spec_info.dataSize      = sizeof(spec_const);
    spec_info.pData         = &spec_const;

    return;
}

bool VkCsExecutor::convolve(const Operation& operation, ShaderConfig& config)
{
#define BUFFER_NUM 4
    opBase->initVulkanThing(BUFFER_NUM);

    const hidl_vec<uint32_t>& ins  = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;

    const size_t inCount = ins.size();
    ASSERT(inCount == 10 || inCount == 7);

    VkOperand& in     = operands[ins[0]];
    VkOperand& filter = operands[ins[1]];
    VkOperand& bias   = operands[ins[2]];
    VkOperand& out    = operands[outs[0]];

    Shape in_shape     = in.getShape();
    Shape out_shape    = out.getShape();
    Shape filter_shape = filter.getShape();
    Shape bias_shape   = bias.getShape();

    int M      = out_shape[kShapeIdxHeight] * out_shape[kShapeIdxWidth];
    int N      = out_shape[kShapeIdxChannel];
    int K      = in_shape[kShapeIdxChannel] * filter_shape[kShapeIdxHeight] * filter_shape[kShapeIdxWidth];
    int tail_m = M % 4;

    PaddingScheme padding_mode;

    SpecializaitonConst spec_const(in_shape[kShapeIdxHeight], in_shape[kShapeIdxWidth],
                                   out_shape[kShapeIdxHeight], out_shape[kShapeIdxWidth],
                                   filter_shape[kShapeIdxHeight], filter_shape[kShapeIdxWidth],
                                   in_shape[kShapeIdxChannel], in_shape[kShapeIdxBatch], M, K, N, tail_m);

    PushConst push_const;

    if (opBase->pipeline == VK_NULL_HANDLE)
    {
        if (inCount == 10)
        {
            uint32_t padding_left   = operands[ins[3]].getScalarData<uint32_t>();
            uint32_t padding_right  = operands[ins[4]].getScalarData<uint32_t>();
            uint32_t padding_top    = operands[ins[5]].getScalarData<uint32_t>();
            uint32_t padding_bottom = operands[ins[6]].getScalarData<uint32_t>();

            spec_const.pad_w        = padding_left;
            spec_const.pad_h        = padding_top;
            spec_const.stride_w     = operands[ins[7]].getScalarData<uint32_t>();
            spec_const.stride_h     = operands[ins[8]].getScalarData<uint32_t>();
            spec_const.activation   = operands[ins[9]].getScalarData<uint32_t>();

            if (padding_left == 0 && padding_top == 0)
            {
                padding_mode = kPaddingValid;
            }
            else
            {
                padding_mode = kPaddingSame;
            }
        }
        else
        {
            padding_mode          = static_cast<PaddingScheme>(operands[ins[3]].getScalarData<uint32_t>());
            spec_const.stride_w   = operands[ins[4]].getScalarData<uint32_t>();
            spec_const.stride_h   = operands[ins[5]].getScalarData<uint32_t>();
            spec_const.activation = operands[ins[6]].getScalarData<uint32_t>();

            calculateExplicitPadding(spec_const.in_w, spec_const.stride_w, spec_const.filter_w,
                                     padding_mode, &spec_const.pad_w);
            calculateExplicitPadding(spec_const.in_h, spec_const.stride_h, spec_const.filter_h,
                                     padding_mode, &spec_const.pad_h);
        }

        // for chn3_to_chn4 convertion
        VkOperand chn4_filter = filter;
        VkOperand chn4_in     = in;

        if (spec_const.channels == 3)
        {
            uint32_t filter_size    = spec_const.channels * spec_const.filter_w * spec_const.filter_h * spec_const.n;
            uint32_t input_size     = spec_const.channels * spec_const.in_w * spec_const.in_h * spec_const.batch;
            uint32_t total_thread_x = 0;

            const int new_channel_num = 4;

            opBase->rebindVkBuffer(chn4_filter, spec_const.n, spec_const.filter_w, spec_const.filter_h, new_channel_num);
            opBase->rebindVkBuffer(chn4_in, spec_const.batch, spec_const.in_w, spec_const.in_h, new_channel_num);

            // first off, convert filter to 4 channels
            if (tmpBoSize == 0)
            {
                total_thread_x = alignSize(filter_size, ITEMS_PER_WI) / ITEMS_PER_WI;
                chn3ToChn4(spec_const, config);

                int group_x = opBase->computeGroupCountX(total_thread_x, spec_const.local_sz_x, spec_const.local_sz_x);
                opBase->setGroupSize(group_x, 1, 1);

                ++tmpBoSize;

                VkSpecializationMapEntry entry[SPEC_CONST_NUM];
                VkSpecializationInfo spec_info;
                spec_const.num_items = filter_size;

                setSpecInfo(entry, spec_info, spec_const, SPEC_CONST_NUM);

                opBase->createShaderModule(conv_chn3to4_spv, sizeof(conv_chn3to4_spv));
                opBase->createPipeline(sizeof(PushConst), &spec_info);

                opBase->bindOperand(filter, 0, opBase->descriptor_set);
                opBase->bindOperand(chn4_filter, 1, opBase->descriptor_set);

                int partition_num = (int)ceil(1.0 * N / opBase->group_y);


                uint32_t num = partition_num * 3;
                for (uint32_t i = 0; i < num; i++)
                {
                    opBase->recordCommandBuffer((void*)&push_const, sizeof(PushConst));
                    opBase->runCommandBuffer();
                }
                // chn4_filter.dumpToFile("filter", 4);
            }

            // then, convert input
            total_thread_x = alignSize(input_size, ITEMS_PER_WI) / ITEMS_PER_WI;
            chn3ToChn4(spec_const, config);

            int group_x = opBase->computeGroupCountX(total_thread_x, spec_const.local_sz_x, spec_const.local_sz_x);
            opBase->setGroupSize(group_x, 1, 1);
            --tmpBoSize;

            VkSpecializationMapEntry entry[SPEC_CONST_NUM];
            VkSpecializationInfo spec_info;

            spec_const.num_items = input_size;
            setSpecInfo(entry, spec_info, spec_const, SPEC_CONST_NUM);

            opBase->createShaderModule(conv_chn3to4_spv, sizeof(conv_chn3to4_spv));
            opBase->createPipeline(sizeof(PushConst), &spec_info);

            opBase->bindOperand(in, 0, opBase->descriptor_set);
            opBase->bindOperand(chn4_in, 1, opBase->descriptor_set);

            int partition_num = (int)ceil(1.0 * N / opBase->group_y);

            for (uint32_t b = 0; b < filter_shape[kShapeIdxBatch]; b++)
            {
                for (int n = 0; n < partition_num; n++)
                {
                    opBase->recordCommandBuffer((void*)&push_const, sizeof(PushConst));
                    opBase->runCommandBuffer();
                }
            }

            spec_const.k = spec_const.k / 3 * 4;
            spec_const.channels = 4;
            converted_to_chn4 = true;

            // chn4_in.dumpToFile("in", 4);
        }

        // prepare shader config
        prepareShaderConfig(spec_const, config);

        spec_const.local_sz_x = config.local_size_x;
        spec_const.local_sz_y = config.local_size_y;
        spec_const.local_sz_z = config.local_size_z;

        VkSpecializationInfo spec_info;
        VkSpecializationMapEntry entry[SPEC_CONST_NUM];
        setSpecInfo(entry, spec_info, spec_const, SPEC_CONST_NUM);

        switch (shader_type)
        {
        case CONV_SHADER_TYPE_GEMM_4_8_GENERIC: {
            opBase->createShaderModule(conv_gemmShader4_8_spv, sizeof(conv_gemmShader4_8_spv));
            opBase->createPipeline(sizeof(PushConst), &spec_info);
            break;
        }
        case CONV_SHADER_TYPE_GEMM1: {
            opBase->createShaderModule(conv_gemm1_spv, sizeof(conv_gemm1_spv));
            opBase->createPipeline(sizeof(PushConst), &spec_info);
            break;
        }
        case CONV_SHADER_TYPE_BASIC:
        case CONV_SHADER_TYPE_GEMM_4_4_NO_IMG2COL:
        case CONV_SHADER_TYPE_GEMM_4_4_GENERIC:
        case CONV_SHADER_TYPE_GEMM_4_4_CHN3: {
            // todo: shaders of gemm_4_4, gemm_no_mig2col and gemm_4_4_chn3 are not added yet
            opBase->createShaderModule(conv_spv, sizeof(conv_spv));
            opBase->createPipeline(sizeof(PushConst), &spec_info);
            break;
        }
        case CONV_SHADER_TYPE_CHN3_TO_CHN4:
        default:
            NOT_REACH_HERE;
            break;
        }

        if (spec_const.channels == 4 && converted_to_chn4)
        {
            // bind the converted channel 4 operands
            opBase->bindOperand(chn4_in, 0, opBase->descriptor_set);
            opBase->bindOperand(chn4_filter, 1, opBase->descriptor_set);
            converted_to_chn4 = false;
        }
        else
        {
            // bind the original input & filter
            opBase->bindOperand(in, 0, opBase->descriptor_set);
            opBase->bindOperand(filter, 1, opBase->descriptor_set);
        }

        // chn3ToChn4 is just for input & filter, no need to convert bias & output
        opBase->bindOperand(bias, 2, opBase->descriptor_set);
        opBase->bindOperand(out, 3, opBase->descriptor_set);
    }

    // todo: should be moved to opBase
    if (false == computeGroupCount(opBase->group_x, opBase->group_y, opBase->group_z, shader_type, spec_const, config))
    {
        NN_GPU_DEBUG("VkCsExecutor::doCONV_2D: computeGroupCount failed");
        return false;
    }
    // todo: duplicated, remove it
    opBase->setGroupSize(opBase->group_x, opBase->group_y, opBase->group_z);

    NN_GPU_DEBUG("VkCsExecutor::doCONV_2D: lsx %d, lsy %d, lsz %d, group_x %d, group_y %d, group_z %d, "
                 "in_w %d, in_h %d, out_h %d, out_w %d, stride_h %d, stride_w %d, pad_h %d, pad_w %d, "
                 "filter_h %d, filter_w %d, channels %d, batch %d, m %d, k %d, n %d, activation %d",
                 spec_const.local_sz_x, spec_const.local_sz_y, spec_const.local_sz_z, opBase->group_x,
                 opBase->group_y, opBase->group_z, spec_const.in_w, spec_const.in_h, spec_const.out_h,
                 spec_const.out_w, spec_const.stride_h, spec_const.stride_w, spec_const.pad_h,
                 spec_const.pad_w, spec_const.filter_h, spec_const.filter_w, spec_const.channels,
                 spec_const.batch, spec_const.m, spec_const.k, spec_const.n, spec_const.activation);

    int partition_num = (int)ceil(1.0 * N / opBase->group_y);

    for (uint32_t b = 0; b < in_shape[kShapeIdxBatch]; b++)
    {
        for (int n = 0; n < partition_num; n++)
        {
            opBase->recordCommandBuffer((void*)&push_const, sizeof(PushConst));
            opBase->runCommandBuffer();
        }
    }

    // out.dumpToFile("out", spec_const.n);
    return true;
}

// FIXME:
// Android NN don't set group, dilation, has_bias,
// so make these assumptions: group = 1, dilation = 1, has_bias = 1
bool VkCsExecutor::doCONV_2D(const Operation& operation)
{
    NN_GPU_ENTRY();

    ASSERT(operation.type == OperationType::CONV_2D);
    bool ret = false;

    ShaderConfig config = {1, 16, 1, 1, 1, 1};
    prepareConfig(operation, config);
    ret = convolve(operation, config);
    if (!ret)
        LOGE("failed to call convolve function");

    NN_GPU_EXIT();
    return ret;
}

NAME_SPACE_STOP
