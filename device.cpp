#include <algorithm>
#include <memory.h>
#include <string.h>

#include <hidl/LegacySupport.h>
#include <thread>

#include "device.h"
#include "prepare_model.h"
#include "executor_manager.h"
#include "ValidateHal.h"

NAME_SPACE_BEGIN

using namespace android::nn;

Return<void> Device::getCapabilities(getCapabilities_cb cb)
{
    NN_GPU_ENTRY();
    NN_GPU_EXIT();

    return Void();
}

Return<void> Device::getCapabilities_1_1(getCapabilities_1_1_cb cb)
{
    NN_GPU_ENTRY();

    Capabilities capabilities;
    ExecutorManager::getCapabilities(capabilities);
    cb(ErrorStatus::NONE, capabilities);

    NN_GPU_EXIT();
    return Void();
}

Return<void> Device::getSupportedOperations(const V10_Model& model,
                                            getSupportedOperations_cb cb)
{
    NN_GPU_ENTRY();
    NN_GPU_EXIT();

    return Void();
}

Return<void> Device::getSupportedOperations_1_1(const Model& model,
                                                getSupportedOperations_1_1_cb cb)
{
    NN_GPU_ENTRY();
    if (!validateModel(model))
    {
        std::vector<bool> supported;
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
        return Void();
    }

    std::vector<bool> supported = ExecutorManager::getSupportedOperations(model);
    cb(ErrorStatus::NONE, supported);
    NN_GPU_EXIT();
    return Void();
}

Return<ErrorStatus> Device::prepareModel(const V10_Model& model,
                                         const sp<IPreparedModelCallback>& callback)
{
    NN_GPU_ENTRY();
    NN_GPU_EXIT();

    return ErrorStatus::NONE;
}

Return<ErrorStatus> Device::prepareModel_1_1(const Model& model,
                                             ExecutionPreference preference,
                                             const sp<IPreparedModelCallback>& callback)
{
    NN_GPU_ENTRY();

    if (callback.get() == nullptr)
    {
        LOGE("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateModel(model) || !validateExecutionPreference(preference))
    {
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    sp<PreparedModel> preparedModel = new PreparedModel(model);
    if (!preparedModel->initialize())
    {
       callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
       return ErrorStatus::INVALID_ARGUMENT;
    }
    callback->notify(ErrorStatus::NONE, preparedModel);

    NN_GPU_EXIT();
    return ErrorStatus::NONE;
}

Return<DeviceStatus> Device::getStatus()
{
    NN_GPU_CALL();
    return DeviceStatus::AVAILABLE;
}

int Device::run()
{
    NN_GPU_CALL();
    if (!ExecutorManager::initPerProcess())
    {
        LOGE("Unable to do ExecutorManager::initPerProcess, service exited!");
        return 1;
    }

    android::hardware::configureRpcThreadpool(4, true);
    if (registerAsService(mName) != android::OK)
    {
        LOGE("Could not register service");
        return 1;
    }
    android::hardware::joinRpcThreadpool();

    LOGE("Service exited!");
    ExecutorManager::deinitPerProcess();

    return 1;
}

NAME_SPACE_STOP
