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

#ifndef ANDROID_HARDWARE_NEURALNETWORKS_V1_0_VK_BUFFER_H
#define ANDROID_HARDWARE_NEURALNETWORKS_V1_0_VK_BUFFER_H

#include <vulkan/vulkan.h>

NAME_SPACE_BEGIN

class Buffer
{
public:
    Buffer(size_t size_in_bytes, const uint8_t* data);
    ~Buffer();
    void dump();
    VkBuffer getVkBuffer() { return buffer; }
    uint8_t* map();
    void unMap();

private:
    Buffer();
    bool init(const uint8_t* data);
    size_t length;
    VkDevice device;
    VkBuffer buffer;
    VkDeviceMemory memory;
};

NAME_SPACE_END

#endif
