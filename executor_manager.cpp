#include "executor_manager.h"
#include "gles/gles_cs_executor.h"
#include "vulkan/vk_cs_executor.h"

NAME_SPACE_BEGIN

ExecutorManager::ExecutorType ExecutorManager::type = ExecutorManager::ET_VK_CS;

bool ExecutorManager::initPerProcess()
{
    NN_GPU_CALL();
    //might check setting to change type

    if (type == ET_GLES_CS)
    {
        return GlesCsExecutor::initPerProcess();
    }
    else if (type == ET_VK_CS)
    {
        return VkCsExecutor::initPerProcess();
    }

    return false;
}

void ExecutorManager::deinitPerProcess()
{
    NN_GPU_ENTRY();
    if (type == ET_GLES_CS)
    {
        GlesCsExecutor::deinitPerProcess();
    }
    else if (type == ET_VK_CS)
    {
        VkCsExecutor::deinitPerProcess();
    }
    NN_GPU_EXIT();
}

void ExecutorManager::getCapabilities(Capabilities &cap)
{
    NN_GPU_ENTRY();
    if (type == ET_GLES_CS)
    {
        GlesCsExecutor::getCapabilities(cap);
    }
    else if (type == ET_VK_CS)
    {
        VkCsExecutor::getCapabilities(cap);
    }
    NN_GPU_EXIT();
}

std::vector<bool> ExecutorManager::getSupportedOperations(const Model& model)
{
    NN_GPU_CALL();
    if (type == ET_GLES_CS)
    {
        return GlesCsExecutor::getSupportedOperations(model);
    }
    else if (type == ET_VK_CS)
    {
        return VkCsExecutor::getSupportedOperations(model);
    }
	else
	{
        const size_t count = model.operations.size();
        std::vector<bool> supported(count, false);
	    return supported;
	}
}

BaseExecutor* ExecutorManager::createExecutor(const Model& model)
{
    NN_GPU_CALL();
    if (type == ET_GLES_CS)
    {
        return new GlesCsExecutor(model);
    }
    else if (type == ET_VK_CS)
    {
        return new VkCsExecutor(model);
    }

	return NULL;
}

NAME_SPACE_STOP