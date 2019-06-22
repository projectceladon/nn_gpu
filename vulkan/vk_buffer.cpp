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

#include "vk_common.h"
#include "vk_buffer.h"
#include "vk_wrapper.h"

NAME_SPACE_BEGIN

static uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memoryProperties;

    vkGetPhysicalDeviceMemoryProperties(kPhysicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((memoryTypeBits & (1 << i)) &&
                ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
            return i;
    }
    return -1;
}

bool Buffer::init(const uint8_t* data)
{
    if (buffer != VK_NULL_HANDLE)
    {
        LOGW("Buffer object already inited\n");
        return false;
    }

    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = length;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, &buffer));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &memory));

    if (data)
    {
        uint8_t* dst;
        VK_CHECK_RESULT(vkMapMemory(device, memory, 0, length, 0, (void **)&dst));
        NN_GPU_DEBUG("call %s, userptr data is %f, size_in_bytes is %zu",
            __func__, 
            *(reinterpret_cast<const float *>(data)),
            length);
        memcpy(dst, data, length);
        vkUnmapMemory(device, memory);
    }
    VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, memory, 0));
    return true;
}

void Buffer::dump()
{
    if (memory != VK_NULL_HANDLE) {
        uint8_t* data;

        VK_CHECK_RESULT(vkMapMemory(device, memory, 0, length, 0, (void **)&data));
        NN_GPU_DEBUG("call %s, userptr data is %f, size_in_bytes is %zu",
            __func__, 
            *(reinterpret_cast<const float *>(data)),
            length);
        // only dump the first 16 bytes
        for (size_t i = 0; i < 15; ++i)
        {
            NN_GPU_DEBUG("dumpped out buffer content is: 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x",
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]);
        }
        vkUnmapMemory(device, memory);
    }
}

Buffer::Buffer(size_t size_in_bytes, const uint8_t* data)
{
    device = kDevice;
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    length = size_in_bytes;
    init(data);
}

Buffer::~Buffer()
{
    vkFreeMemory(device, memory, NULL);
    vkDestroyBuffer(device, buffer, NULL);
}

uint8_t* Buffer::map()
{
    void *p;
    ASSERT(memory != VK_NULL_HANDLE);

    VK_CHECK_RESULT(vkMapMemory(device, memory,
                                0, length, 0, (void **)&p));

    return (uint8_t*)p;
}

void Buffer::unMap()
{
    vkUnmapMemory(device, memory);
}

NAME_SPACE_STOP
