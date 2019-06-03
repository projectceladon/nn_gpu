/*
 * Copyright @2019 Intel Corporation
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

// todo: query group count from vulkan device
#define MAX_GROUP_COUNT_X 65535
#define MAX_GROUP_COUNT_Y 65535
#define MAX_GROUP_COUNT_Z 65535

#define DEFAULT_DILATION_H 1
#define DEFAULT_DILATION_W 1
#define HAS_BIAS 1


struct DepthWiseParameter
{
    DepthWiseParameter(int ih, int iw, int oh, int ow, int dh, int dw, int fh, int fw, int chn,
                       int bias, int M, int K, int N):
        in_h(ih), in_w(iw), out_h(oh), out_w(ow), dilation_h(dh), dilation_w(dw), filter_h(fh), filter_w(fw),
        channels(chn), has_bias(bias), m(M), k(K), n(N)
    {};

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
    int has_bias;
    int m;
    int k;
    int n;
    int depth_multiplier;
    int activation;
};

static void prepareConfig(const Operation& operation, ShaderConfig& config)
{
    //tune();
    (void)(operation);
    (void)(config);
}

bool VkCsExecutor::depthConvolve(const Operation& operation, ShaderConfig& config)
{
#define BUFFER_NUM 4
    opBase->initVulkanThing(BUFFER_NUM);

    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;

    ASSERT(ins.size() == 11 || ins.size() == 8);

    VkOperand& in     = operands[ins[0]];
    VkOperand& filter = operands[ins[1]];
    VkOperand& bias   = operands[ins[2]];
    VkOperand& out    = operands[outs[0]];

    Shape in_shape     = in.getShape();
    Shape out_shape    = out.getShape();
    Shape filter_shape = filter.getShape();
    Shape bias_shape   = bias.getShape();

    uint32_t M = out_shape[kShapeIdxHeight] * out_shape[kShapeIdxWidth];
    uint32_t N = out_shape[kShapeIdxChannel];
    uint32_t K = in_shape[kShapeIdxChannel] * filter_shape[kShapeIdxHeight] * filter_shape[kShapeIdxWidth];

    PaddingScheme padding_mode;
    DepthWiseParameter param(in_shape[kShapeIdxHeight], in_shape[kShapeIdxWidth],
                             out_shape[kShapeIdxHeight], out_shape[kShapeIdxWidth],
                             DEFAULT_DILATION_H, DEFAULT_DILATION_W,
                             filter_shape[kShapeIdxHeight], filter_shape[kShapeIdxWidth],
                             in_shape[kShapeIdxChannel], HAS_BIAS, M, K, N);

    if (opBase->pipeline == VK_NULL_HANDLE)
    {
        if (ins.size() == 11)
        {
            uint32_t padding_left    = operands[ins[3]].getScalarData<uint32_t>();
            uint32_t padding_right   = operands[ins[4]].getScalarData<uint32_t>();
            uint32_t padding_top     = operands[ins[5]].getScalarData<uint32_t>();
            uint32_t padding_bottom  = operands[ins[6]].getScalarData<uint32_t>();
            // todo: should be add or minus.
            param.pad_w              = padding_right;
            param.pad_h              = padding_bottom;
            param.stride_w           = operands[ins[7]].getScalarData<uint32_t>();
            param.stride_h           = operands[ins[8]].getScalarData<uint32_t>();
            param.depth_multiplier   = operands[ins[9]].getScalarData<uint32_t>();
            param.activation         = operands[ins[10]].getScalarData<uint32_t>();

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
            padding_mode           = static_cast<PaddingScheme>(operands[ins[3]].getScalarData<uint32_t>());
            param.stride_w         = operands[ins[4]].getScalarData<uint32_t>();
            param.stride_h         = operands[ins[5]].getScalarData<uint32_t>();
            param.depth_multiplier = operands[ins[6]].getScalarData<uint32_t>();
            param.activation       = operands[ins[7]].getScalarData<uint32_t>();

            calculateExplicitPadding(param.in_w,
                    param.stride_w, param.filter_w, padding_mode, &param.pad_w);
            calculateExplicitPadding(param.in_h,
                    param.stride_h, param.filter_h, padding_mode, &param.pad_h);
        }

        NN_GPU_DEBUG("run createShaderModule");
        opBase->createShaderModule(dw_conv_spv, sizeof(dw_conv_spv));

        NN_GPU_DEBUG("run createPipeline");
        opBase->createPipeline(sizeof(DepthWiseParameter));
    }

    opBase->group_x = ceil(static_cast<float>(param.out_w) / config.local_size_x);
    opBase->group_y = ceil(static_cast<float>(param.out_h) / config.local_size_y);
    opBase->group_z = ceil(static_cast<float>
          ((ceil(static_cast<float>(N) * in_shape[kShapeIdxBatch] / param.depth_multiplier))) / config.local_size_z);

    NN_GPU_DEBUG("bind operands");
    opBase->bindOperand(in, 0, opBase->descriptor_set);
    opBase->bindOperand(filter, 1, opBase->descriptor_set);
    opBase->bindOperand(bias, 2, opBase->descriptor_set);
    opBase->bindOperand(out, 3, opBase->descriptor_set);

    int partition_num = 1;
    partition_num = (int)ceil(1.0 * N / opBase->group_y);

    for (uint32_t b = 0;  b < in_shape[kShapeIdxBatch]; b++)
    {
        for (int n = 0;  n < partition_num; n++)
        {
            NN_GPU_DEBUG("run recordCommandBuffer");
            opBase->recordCommandBuffer((void *)&param, sizeof(DepthWiseParameter));

            NN_GPU_DEBUG("run runCommandBuffer");
            opBase->runCommandBuffer();
        }
    }

    return true;
}

bool VkCsExecutor::doDEPTHWISE_CONV_2D(const Operation& operation)
{
    NN_GPU_ENTRY();

    ASSERT(operation.type == OperationType::DEPTHWISE_CONV_2D);

    bool ret = false;

    // config: auto config.
    ShaderConfig config = {1, 1, 16, 1, 1, 1};
    prepareConfig(operation, config);

    ret = depthConvolve(operation, config);

    NN_GPU_EXIT();

    return ret;
}

NAME_SPACE_STOP
