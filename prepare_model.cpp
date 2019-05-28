#include <algorithm>
#include <memory.h>
#include <string.h>

#include <hidl/LegacySupport.h>
#include <thread>

#include "prepare_model.h"
#include "executor_manager.h"
#include "validate.h"

NAME_SPACE_BEGIN

PreparedModel::PreparedModel(const Model& model)
      : // Make a copy of the model, as we need to preserve it.
        mModel(model)
{
    NN_GPU_CALL();
    exec = ExecutorManager::createExecutor(mModel);
}

bool PreparedModel::initialize()
{
    NN_GPU_CALL();
    exec->initPerModel();
    return true;
}

void PreparedModel::asyncExecute(const Request& request,
                                       const sp<IExecutionCallback>& callback)
{
    NN_GPU_CALL();
    exec->initPerExecThread();
    bool succ = exec->run(request);
    exec->deinitPerExecThread();
    if (succ)
    {
        callback->notify(ErrorStatus::NONE);
    }
    else
    {
        callback->notify(ErrorStatus::GENERAL_FAILURE);
    }
}

Return<ErrorStatus> PreparedModel::execute(const Request& request,
                                                 const sp<IExecutionCallback>& callback)
{
    NN_GPU_CALL();
    if (callback.get() == nullptr)
    {
        LOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, mModel)) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    execThreads.push_back(std::thread([this, request, callback]{ asyncExecute(request, callback); }));

    return ErrorStatus::NONE;
}

PreparedModel::~PreparedModel()
{
    NN_GPU_CALL();
    for (auto& th : execThreads) th.join();
    exec->deinitPerModel();
}

NAME_SPACE_END
