#ifndef ANDROID_HARDWARE_NEURALNETWORKS_V1_0_DEVICE_H
#define ANDROID_HARDWARE_NEURALNETWORKS_V1_0_DEVICE_H

#include "hal_types.h"

NAME_SPACE_BEGIN

class Device : public IDevice {
public:
    Device(const char* name) : mName(name) {}
    ~Device() override {}
    Return<void> getCapabilities(getCapabilities_cb _hidl_cb) override;
    Return<void> getSupportedOperations(const Model& model, getSupportedOperations_cb cb) override;

    Return<ErrorStatus> prepareModel(const Model& model,
                                     const sp<IPreparedModelCallback>& callback) override;
    Return<DeviceStatus> getStatus() override;

    // Starts and runs the driver service.  Typically called from main().
    // This will return only once the service shuts down.
    int run();
protected:
    std::string mName;
};

NAME_SPACE_END

#endif