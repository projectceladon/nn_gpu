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

struct SpecializationConst {
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
};

struct PushConst {
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

    VkOperand& in  = operands[ins[0]];
    VkOperand& filter = operands[ins[1]];
    VkOperand& bias   = operands[ins[2]];
    VkOperand& out = operands[outs[0]];

	Shape in_shape = in.getShape();
	Shape out_shape = out.getShape();
	Shape filter_shape = filter.getShape();
	Shape bias_shape = bias.getShape();

    int M = out_shape[kShapeIdxHeight] * out_shape[kShapeIdxWidth];
    int N = out_shape[kShapeIdxChannel];
    int K = in_shape[kShapeIdxChannel] * filter_shape[kShapeIdxHeight] * filter_shape[kShapeIdxWidth];

	uint32_t out_channel = out_shape[kShapeIdxChannel];
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

	if (opBase->pipeline == VK_NULL_HANDLE)
	{
        // specialization constants
        VkSpecializationInfo spec_info;
        SpecializationConst spec_const;
        VkSpecializationMapEntry entry[SPECIALIZATION_CONST_NUM];

        spec_const.lsz_x = config.local_size_x;
        spec_const.lsz_y = config.local_size_y;
        spec_const.lsz_z = config.local_size_z;
        spec_const.in_h  = in_shape[kShapeIdxHeight];
        spec_const.in_w  = in_shape[kShapeIdxWidth];
        spec_const.out_h = out_shape[kShapeIdxHeight];
        spec_const.out_w = out_shape[kShapeIdxWidth];
        spec_const.dilation_h = 1;
        spec_const.dilation_w = 1;
        spec_const.filter_h = filter_shape[kShapeIdxHeight];
        spec_const.filter_w = filter_shape[kShapeIdxWidth];
        spec_const.channels = in_shape[kShapeIdxChannel];
        spec_const.batch = in_shape[kShapeIdxBatch];
        spec_const.has_bias = 1;
        spec_const.m = M;
        spec_const.k = K;
        spec_const.n = N;
        spec_const.tail_m = M % 4;
        spec_const.depth_multiplier = DEFAULT_DEPTH_MULTIPLIER;

        if (inCount == 10) {
            uint32_t padding_left   = operands[ins[3]].getScalarData<uint32_t>();
            uint32_t padding_right  = operands[ins[4]].getScalarData<uint32_t>();
            uint32_t padding_top    = operands[ins[5]].getScalarData<uint32_t>();
            uint32_t padding_bottom = operands[ins[6]].getScalarData<uint32_t>();
            spec_const.pad_w        = padding_left + padding_right;
            spec_const.pad_h        = padding_top + padding_bottom;
            spec_const.stride_w     = operands[ins[7]].getScalarData<uint32_t>();
            spec_const.stride_h     = operands[ins[8]].getScalarData<uint32_t>();
            //it is not used in compute shader, only support case of activtion as 0
            spec_const.activation   = operands[ins[9]].getScalarData<uint32_t>();
            assert(spec_const.activation == 0);
            if (padding_left == 0 && padding_top == 0)
                padding_mode = kPaddingValid;
            else
                padding_mode = kPaddingSame;
        } else {
            padding_mode = static_cast<PaddingScheme>(operands[ins[3]].getScalarData<uint32_t>());
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

        SET_SPEC_CONST_ENTRY(entry[0], 0, offsetof(SpecializationConst,lsz_x), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[1], 1, offsetof(SpecializationConst,lsz_y), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[2], 2, offsetof(SpecializationConst,lsz_z), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[3], 3, offsetof(SpecializationConst,in_h), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[4], 4, offsetof(SpecializationConst,in_w), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[5], 5, offsetof(SpecializationConst,out_h), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[6], 6, offsetof(SpecializationConst,out_w), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[7], 7, offsetof(SpecializationConst,stride_h), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[8], 8, offsetof(SpecializationConst,stride_w), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[9], 9, offsetof(SpecializationConst,dilation_h), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[10], 10, offsetof(SpecializationConst,dilation_w), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[11], 11, offsetof(SpecializationConst,pad_h), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[12], 12, offsetof(SpecializationConst,pad_w), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[13], 13, offsetof(SpecializationConst,filter_h), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[14], 14, offsetof(SpecializationConst,filter_w), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[15], 15, offsetof(SpecializationConst,channels), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[16], 16, offsetof(SpecializationConst,batch), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[17], 17, offsetof(SpecializationConst,has_bias), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[18], 18, offsetof(SpecializationConst,m), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[19], 19, offsetof(SpecializationConst,k), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[20], 20, offsetof(SpecializationConst,n), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[21], 21, offsetof(SpecializationConst,tail_m), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[22], 22, offsetof(SpecializationConst,depth_multiplier), sizeof(int));
        SET_SPEC_CONST_ENTRY(entry[23], 23, offsetof(SpecializationConst,activation), sizeof(int));
        spec_info.mapEntryCount = SPECIALIZATION_CONST_NUM;
        spec_info.pMapEntries   = entry;
        spec_info.dataSize      = sizeof(spec_const);
        spec_info.pData         = &spec_const;

		opBase->createShaderModule(conv_spv, sizeof(conv_spv));
		opBase->createPipeline(sizeof(PushConst), &spec_info);
	}

    opBase->group_x = alignSize(M, config.local_size_x) / config.local_size_x;
    float GFLOPS = (2.0 * K + 1) * M * N / 1000 / 1000 / 1000;
    assert(config.local_size_y == 1);
    opBase->group_y = std::min(MAX_GROUP_COUNT_Y, (int)floor(MAX_COMPUTE_GFLOPS / (GFLOPS / out_channel)));
    opBase->group_z = 1;

	opBase->bindOperand(in, 0, opBase->descriptor_set);
	opBase->bindOperand(bias, 1, opBase->descriptor_set);
	opBase->bindOperand(filter, 2, opBase->descriptor_set);
	opBase->bindOperand(out, 3, opBase->descriptor_set);

    PushConst param;
    int partition_num = 1;
    param.basic_shader_partition_size = opBase->group_y;
    partition_num = (int)ceil(1.0 * out_channel / opBase->group_y);

    for (uint32_t b = 0;  b < in_shape[kShapeIdxBatch]; b++)
    {
        param.basic_shader_batch_idx = b;
        for (int n = 0;  n < partition_num; n++)
        {
            param.basic_shader_partition_idx = n;
            opBase->recordCommandBuffer((void *)&param, sizeof(PushConst));
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

    ShaderConfig config = {DEFAULT_LOCAL_SZ, 1, 1, 1, 1, 1};
    prepareConfig(operation, config);
    return convolve(operation, config);
}

NAME_SPACE_STOP
