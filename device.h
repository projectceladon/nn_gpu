#ifndef ANDROID_HARDWARE_NEURALNETWORKS_V1_1_DEVICE_H
#define ANDROID_HARDWARE_NEURALNETWORKS_V1_1_DEVICE_H

#include "hal_types.h"

NAME_SPACE_BEGIN

class Device : public IDevice {
public:
    Device(const char* name) : mName(name) {}
    virtual ~Device() override {}
    virtual Return<void> getCapabilities(getCapabilities_cb _hidl_cb) override;
    virtual Return<void> getCapabilities_1_1(getCapabilities_1_1_cb cb) override;
    virtual Return<void> getSupportedOperations(const V10_Model& model, getSupportedOperations_cb cb) override;
    virtual Return<void> getSupportedOperations_1_1(const Model& model,
                                            getSupportedOperations_1_1_cb cb) override;
    virtual Return<ErrorStatus> prepareModel(const V10_Model& model,
                                     const sp<IPreparedModelCallback>& callback) override;
    virtual Return<ErrorStatus> prepareModel_1_1(const Model& model, ExecutionPreference preference,
                                         const sp<IPreparedModelCallback>& callback) override;
    virtual Return<DeviceStatus> getStatus() override;

    // Starts and runs the driver service.  Typically called from main().
    // This will return only once the service shuts down.
    int run();
protected:
    std::string mName;
};

NAME_SPACE_STOP

#endif
