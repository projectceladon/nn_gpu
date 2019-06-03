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

#ifndef ANDROID_HARDWARE_NEURALNETWORKS_V1_0_VK_SPV_SHADER_H
#define ANDROID_HARDWARE_NEURALNETWORKS_V1_0_VK_SPV_SHADER_H

NAME_SPACE_BEGIN

extern const unsigned int elewise_spv[890];
extern const unsigned int conv_spv[1739];
extern const unsigned int concat_spv[541];
extern const unsigned int softmax_spv[900];
extern const unsigned int avg_pool_spv[1538];
extern const unsigned int max_pool_spv[1449];
extern const unsigned int lrn_spv[1730];
extern const unsigned int dw_conv_spv[2292];
extern const unsigned int logistic_spv[368];

NAME_SPACE_STOP

#endif
