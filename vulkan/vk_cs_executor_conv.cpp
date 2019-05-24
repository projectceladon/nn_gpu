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

#define DEFAULT_LOCAL_SZ 256
#define SPECIALIZATION_CONST_NUM 24
#define DEFAULT_DEPTH_MULTIPLIER 1
#define MAX_COMPUTE_GFLOPS 10
// TODO: query group count from vulkan device
#define MAX_GROUP_COUNT_X 65535
#define MAX_GROUP_COUNT_Y 65535
#define MAX_GROUP_COUNT_Z 65535
#define LOCAL_SZ_X 256

#define DEFAULT_DILATION_H 1
#define DEFAULT_DILATION_W 1
#define HAS_BIAS 1

struct SpecializationConst {
public:
    SpecializationConst(int ih, int iw, int oh, int ow, int dh, int dw, int fh, int fw, int chn,
                        int bat, int bias, int M, int K, int N, int tm, int dm):
        in_h(ih), in_w(iw), out_h(oh), out_w(ow), dilation_h(dh), dilation_w(dw), filter_h(fh), filter_w(fw),
        channels(chn), batch(bat), has_bias(bias), m(M), k(K), n(N), tail_m(tm), depth_multiplier(dm)
    {};

    int lsz_x;
    int lsz_y;
    int lsz_z;
    int in_h;
    int in_w;
    int out_h;
    int out_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int pad_h;
    int pad_w;
    int filter_h;
    int filter_w;
    int channels;
    int batch;
    int has_bias;
    int m;
    int k;
    int n;
    int tail_m;
    int depth_multiplier;
    int activation;
    int basic_shader_batch_idx;
    int basic_shader_partition_idx;
    int basic_shader_partition_size;
};

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

bool VkCsExecutor::convolve(const Operation& operation, ShaderConfig& config)
{
    const hidl_vec<uint32_t>& ins = operation.inputs;
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

    int M = out_shape[kShapeIdxHeight] * out_shape[kShapeIdxWidth];
    int N = out_shape[kShapeIdxChannel];
    int K = in_shape[kShapeIdxChannel] * filter_shape[kShapeIdxHeight] * filter_shape[kShapeIdxWidth];

    PaddingScheme padding_mode;

#if 0
	uint32_t reshape_out_h, reshape_out_w;
	computeConvOutputShapeAndPadding(padding_mode,
									 param.in_h, param.in_w,
									 param.filter_h, param.filter_w,
									 param.dilation_h, param.dilation_w,
									 param.stride_h, param.stride_w,
									 reshape_out_h, reshape_out_w);
	Shape shape = {param.batch, out_channel, reshape_out_h, reshape_out_w};
	//out.reshape(NULL, shape);
#endif

    // specialization constants
    SpecializationConst spec_const(in_shape[kShapeIdxHeight], in_shape[kShapeIdxWidth],
                                   out_shape[kShapeIdxHeight], out_shape[kShapeIdxWidth],
                                   DEFAULT_DILATION_H, DEFAULT_DILATION_W,
                                   filter_shape[kShapeIdxHeight], filter_shape[kShapeIdxWidth],
                                   in_shape[kShapeIdxChannel], in_shape[kShapeIdxBatch], HAS_BIAS, M, K, N, M % 4,
                                   DEFAULT_DEPTH_MULTIPLIER);

    if (opBase->pipeline == VK_NULL_HANDLE)
    {
        if (inCount == 10)
        {
            uint32_t padding_left   = operands[ins[3]].getScalarData<uint32_t>();
            uint32_t padding_right  = operands[ins[4]].getScalarData<uint32_t>();
            uint32_t padding_top    = operands[ins[5]].getScalarData<uint32_t>();
            uint32_t padding_bottom = operands[ins[6]].getScalarData<uint32_t>();

            spec_const.pad_w        = padding_left - padding_right;
            spec_const.pad_h        = padding_top - padding_bottom;
            spec_const.stride_w     = operands[ins[7]].getScalarData<uint32_t>();
            spec_const.stride_h     = operands[ins[8]].getScalarData<uint32_t>();
            spec_const.activation   = operands[ins[9]].getScalarData<uint32_t>();

            assert(spec_const.activation == 0);  // TODO: add activation

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
            padding_mode            = static_cast<PaddingScheme>(operands[ins[3]].getScalarData<uint32_t>());
            spec_const.stride_w     = operands[ins[4]].getScalarData<uint32_t>();
            spec_const.stride_h     = operands[ins[5]].getScalarData<uint32_t>();
            spec_const.activation   = operands[ins[6]].getScalarData<uint32_t>();

            assert(spec_const.activation == 0);

            calculateExplicitPadding(spec_const.in_w, spec_const.stride_w,
                                     spec_const.filter_w, padding_mode,
                                     &spec_const.pad_w);
            calculateExplicitPadding(spec_const.in_h, spec_const.stride_h,
                                     spec_const.filter_h, padding_mode,
                                     &spec_const.pad_h);
        }

        NN_GPU_DEBUG("run createShaderModule");
        opBase->createShaderModule(conv_spv, sizeof(conv_spv));

        NN_GPU_DEBUG("run createPipeline");
        opBase->createPipeline(sizeof(SpecializationConst));
    }

    opBase->group_x = alignSize(alignSize(N, config.block_width) / config.block_width, config.local_size_x) / config.local_size_x;
    opBase->group_y = alignSize(alignSize(M, config.block_height) / config.block_height, config.local_size_y) / config.local_size_y;
    opBase->group_z = alignSize(alignSize(spec_const.batch, config.block_depth), config.local_size_z) / config.local_size_z;

    NN_GPU_DEBUG("bind operands");
    opBase->bindOperand(in, 0, opBase->descriptor_set);
    opBase->bindOperand(filter, 1, opBase->descriptor_set);
    opBase->bindOperand(bias, 2, opBase->descriptor_set);
    opBase->bindOperand(out, 3, opBase->descriptor_set);

    int partition_num = 1;
    spec_const.basic_shader_partition_size = opBase->group_y;
    partition_num = (int)ceil(1.0 * N / opBase->group_y);

    for (uint32_t b = 0;  b < in_shape[kShapeIdxBatch]; b++)
    {
        spec_const.basic_shader_batch_idx = b;
        for (int n = 0;  n < partition_num; n++)
        {
            spec_const.basic_shader_partition_idx = n;
            opBase->recordCommandBuffer((void *)&spec_const, sizeof(SpecializationConst));
            opBase->runCommandBuffer();
        }
    }

    return true;
}

// FIXME:
// Android NN don't set group, dilation, has_bias,
// so make these assumptions: group = 1, dilation = 1, has_bias = 1
bool VkCsExecutor::doCONV_2D(const Operation& operation)
{
    ASSERT(operation.type == OperationType::CONV_2D);

#define BUFFER_NUM 2
    opBase->initVulkanThing(BUFFER_NUM);

    ShaderConfig config = {1, 16, 1, 1, 1, 1};
    prepareConfig(operation, config);
    return convolve(operation, config);
}

NAME_SPACE_STOP
