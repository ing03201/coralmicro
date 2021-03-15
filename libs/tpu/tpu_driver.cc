#include "libs/tpu/tpu_driver.h"
#include "third_party/darwinn/driver/config/beagle/beagle_chip_config.h"
#include "third_party/darwinn/driver/config/beagle_csr_helper.h"
#include "third_party/darwinn/driver/config/common_csr_helper.h"
#include "third_party/nxp/rt1176-sdk/components/osa/fsl_os_abstraction.h"
#include "third_party/nxp/rt1176-sdk/middleware/usb/include/usb_spec.h"

#include <cassert>

constexpr uint32_t kMaxBulkBufferSize = 32 * 1024;
static uint8_t BulkTransferBuffer[kMaxBulkBufferSize];

namespace valiant {

namespace registers = platforms::darwinn::driver::config::registers;

TpuDriver::TpuDriver() {
}

bool TpuDriver::Initialize(usb_host_edgetpu_instance_t *usb_instance) {
    if (usb_instance == nullptr) {
        return false;
    }
    usb_instance_ = usb_instance;

    // Check chip id and test write
    uint32_t omc0_00_reg;
    assert(Read32(chip_config_.GetApexCsrOffsets().omc0_00, &omc0_00_reg));
    registers::Omc000 omc0_00(omc0_00_reg);
    assert(0x89A == omc0_00.chip_id());

    omc0_00.set_test_reg0(0xAA);
    assert(Write32(chip_config_.GetApexCsrOffsets().omc0_00, omc0_00.raw()));

    omc0_00_reg = 0;
    assert(Read32(chip_config_.GetApexCsrOffsets().omc0_00, &omc0_00_reg));
    omc0_00.set_raw(omc0_00_reg);
    assert(0xAA == omc0_00.test_reg0());

    // Disable inactive mode
    uint32_t scu_ctrl_0_reg;
    assert(Read32(chip_config_.GetScuCsrOffsets().scu_ctrl_0, &scu_ctrl_0_reg));
    registers::ScuCtrl0 scu_ctrl_0(scu_ctrl_0_reg);
    scu_ctrl_0.set_rg_pcie_inact_phy_mode(0);
    scu_ctrl_0.set_rg_usb_inact_phy_mode(0);
    assert(Write32(chip_config_.GetScuCsrOffsets().scu_ctrl_0, scu_ctrl_0.raw()));
    assert(Read32(chip_config_.GetScuCsrOffsets().scu_ctrl_0, &scu_ctrl_0_reg));

    // Disable clock gating
    uint32_t scu_ctrl_2_reg;
    assert(Read32(chip_config_.GetScuCsrOffsets().scu_ctrl_2, &scu_ctrl_2_reg));
    registers::ScuCtrl2 scu_ctrl_2(scu_ctrl_2_reg);
    scu_ctrl_2.set_rg_gated_gcb(0x2);
    assert(Write32(chip_config_.GetScuCsrOffsets().scu_ctrl_2, scu_ctrl_2.raw()));
    assert(Read32(chip_config_.GetScuCsrOffsets().scu_ctrl_2, &scu_ctrl_2_reg));

    // Go into reset, if we're not there
    uint32_t scu_ctrl_3_reg;
    assert(Read32(chip_config_.GetScuCsrOffsets().scu_ctrl_3, &scu_ctrl_3_reg));
    registers::ScuCtrl3 scu_ctrl_3(scu_ctrl_3_reg);
    if (scu_ctrl_3.rg_force_sleep() != 0x3) {
        scu_ctrl_3.set_rg_force_sleep(0x3);
        assert(Write32(chip_config_.GetScuCsrOffsets().scu_ctrl_3, scu_ctrl_3.raw()));
        do {
            assert(Read32(chip_config_.GetScuCsrOffsets().scu_ctrl_3, &scu_ctrl_3_reg));
            scu_ctrl_3.set_raw(scu_ctrl_3_reg);
        } while (scu_ctrl_3.cur_pwr_state() != 0x2);
        assert(Write32(chip_config_.GetCbBridgeCsrOffsets().gcbb_credit0, 0xF));
        assert(Write32(chip_config_.GetCbBridgeCsrOffsets().gcbb_credit0, 0x0));
    }

    // TODO(atv): Support something besides MAX
    // Set performance mode and exit reset.
    assert(Read32(chip_config_.GetScuCsrOffsets().scu_ctrl_3, &scu_ctrl_3_reg));
    scu_ctrl_3.set_raw(scu_ctrl_3_reg);
    scu_ctrl_3.set_rg_force_sleep(0x2);
    // Max
    // scu_ctrl_3.set_gcb_clock_rate(registers::ScuCtrl3::GcbClock::k500MHZ);
    // scu_ctrl_3.set_axi_clock_rate(registers::ScuCtrl3::AxiClock::k250MHZ);
    // scu_ctrl_3.set_usb_8051_clock_rate(registers::ScuCtrl3::Usb8051Clock::k500MHZ);
    // High
    scu_ctrl_3.set_gcb_clock_rate(registers::ScuCtrl3::GcbClock::k250MHZ);
    scu_ctrl_3.set_axi_clock_rate(registers::ScuCtrl3::AxiClock::k125MHZ);
    scu_ctrl_3.set_usb_8051_clock_rate(registers::ScuCtrl3::Usb8051Clock::k500MHZ);
    assert(Write32(chip_config_.GetScuCsrOffsets().scu_ctrl_3, scu_ctrl_3.raw()));

    do {
        assert(Read32(chip_config_.GetScuCsrOffsets().scu_ctrl_3, &scu_ctrl_3_reg));
        scu_ctrl_3.set_raw(scu_ctrl_3_reg);
    } while (scu_ctrl_3.cur_pwr_state() != 0x0);

    // Check a known register to verify reset exit.
    uint64_t scalar_core_run_control;
    do {
        assert(Read64(chip_config_.GetScalarCoreCsrOffsets().scalarCoreRunControl, &scalar_core_run_control));
    } while (scalar_core_run_control != 0);

    registers::IdleRegister idle_reg;
    idle_reg.set_enable();
    idle_reg.set_counter(1);
    assert(Write64(chip_config_.GetMiscCsrOffsets().idleRegister, idle_reg.raw()));

    registers::TileConfig<7> tile_config;
    tile_config.set_broadcast();
    assert(Write64(chip_config_.GetTileConfigCsrOffsets().tileconfig0, tile_config.raw()));

    uint64_t tile_config_reg;
    do {
        assert(Read64(chip_config_.GetTileConfigCsrOffsets().tileconfig0, &tile_config_reg));
    } while(tile_config.raw() != tile_config_reg);

    registers::DeepSleep deep_sleep_reg;
    deep_sleep_reg.set_to_sleep_delay(2);
    deep_sleep_reg.set_to_wake_delay(30);
    assert(Write64(chip_config_.GetTileCsrOffsets().deepSleep, deep_sleep_reg.raw()));

    // Enable clock gating
    assert(Read32(chip_config_.GetScuCsrOffsets().scu_ctrl_2, &scu_ctrl_2_reg));
    scu_ctrl_2.set_raw(scu_ctrl_2_reg);
    scu_ctrl_2.set_rg_gated_gcb(1);
    assert(Write32(chip_config_.GetScuCsrOffsets().scu_ctrl_2, scu_ctrl_2.raw()));

    assert(Write64(chip_config_.GetUsbCsrOffsets().descr_ep, 0xF0));
    assert(Write64(chip_config_.GetUsbCsrOffsets().multi_bo_ep, 0));
    assert(Write64(chip_config_.GetUsbCsrOffsets().outfeed_chunk_length, 0x20));

    assert(DoRunControl(platforms::darwinn::driver::RunControl::kMoveToRun));

    return true;
}

bool TpuDriver::CSRTransfer(uint64_t reg, void *data, bool read, RegisterSize reg_size) {
    bool ret = false;
    usb_status_t control_status;
    usb_setup_struct_t setup_packet;
    setup_packet.bmRequestType = USB_REQUEST_TYPE_TYPE_VENDOR | USB_REQUEST_TYPE_RECIPIENT_DEVICE;
    setup_packet.bmRequestType |= read ? USB_REQUEST_TYPE_DIR_IN : USB_REQUEST_TYPE_DIR_OUT;
    switch (reg_size) {
        case RegisterSize::kRegSize32:
            setup_packet.bRequest = 1;
            setup_packet.wLength = 4;
            break;
        case RegisterSize::kRegSize64:
            setup_packet.bRequest = 0;
            setup_packet.wLength = 8;
            break;
    }

    setup_packet.wValue = 0xFFFF & reg;
    setup_packet.wIndex = 0xFFFF & (reg >> 16);

    SemaphoreHandle_t sema = xSemaphoreCreateBinary();

    control_status = USB_HostEdgeTpuControl(usb_instance_, &setup_packet, (uint8_t*)data,
        [](void *param, uint8_t *data, uint32_t data_length, usb_status_t status) {
            SemaphoreHandle_t sema = (SemaphoreHandle_t)param;
            xSemaphoreGive(sema);
        }, sema);
    if (control_status != kStatus_USB_Success) {
        printf("USB_HostEdgeTpuControl failed\r\n");
        goto exit;
    }
    if (xSemaphoreTake(sema, pdMS_TO_TICKS(200)) == pdFALSE) {
        ret = false;
        printf("%s didn't get semaphore\r\n", __func__);
        goto exit;
    }

    ret = true;
exit:
    vSemaphoreDelete(sema);
    return ret;
}

bool TpuDriver::SendData(DescriptorTag tag, const uint8_t *data, uint32_t length) const {
    if (!WriteHeader(tag, length)) {
        printf("WriteHeader failed\r\n");
        return false;
    }

    if (!BulkOutTransfer(data, length)) {
        printf("BulkOutTransfer failed\r\n");
        return false;
    }
    return true;
}

bool TpuDriver::SendParameters(const uint8_t *data, uint32_t length) const {
    return SendData(DescriptorTag::kParameters, data, length);
}

bool TpuDriver::SendInputs(const uint8_t *data, uint32_t length) const {
    return SendData(DescriptorTag::kInputActivations, data, length);
}

bool TpuDriver::SendInstructions(const uint8_t *data, uint32_t length) const {
    return SendData(DescriptorTag::kInstructions, data, length);
}

bool TpuDriver::GetOutputs(uint8_t *data, uint32_t length) const {
    return BulkInTransfer(data, length);
}

bool TpuDriver::Read32(uint64_t reg, uint32_t *val) {
    return CSRTransfer(reg, val, true, RegisterSize::kRegSize32);
}

bool TpuDriver::Read64(uint64_t reg, uint64_t *val) {
    return CSRTransfer(reg, val, true, RegisterSize::kRegSize64);
}

bool TpuDriver::Write32(uint64_t reg, uint32_t val) {
    return CSRTransfer(reg, &val, false, RegisterSize::kRegSize32);
}

bool TpuDriver::Write64(uint64_t reg, uint64_t val) {
    return CSRTransfer(reg, &val, false, RegisterSize::kRegSize64);
}

bool TpuDriver::BulkOutTransferInternal(uint8_t endpoint, const uint8_t *data, uint32_t data_length) const {
    bool ret = false;
    SemaphoreHandle_t sema = xSemaphoreCreateBinary();

    usb_status_t bulk_status = USB_HostEdgeTpuBulkOutSend(usb_instance_, endpoint,
            (uint8_t*)data, data_length,
            [](void *param,
               uint8_t *data,
               uint32_t data_length,
               usb_status_t status) {
                SemaphoreHandle_t sema = (SemaphoreHandle_t)param;
                xSemaphoreGive(sema);
            }, sema);

    if (bulk_status != kStatus_USB_Success) {
        printf("USB_HostEdgeTpuBulkOutSend failed\r\n");
        goto exit;
    }

    if (xSemaphoreTake(sema, pdMS_TO_TICKS(200)) == pdFALSE) {
        printf("%s didn't get semaphore\r\n", __func__);
        goto exit;
    };

    ret = true;
exit:
    vSemaphoreDelete(sema);
    return ret;
}

bool TpuDriver::BulkOutTransfer(const uint8_t *data, uint32_t data_length) const {
    uint8_t *current_chunk = const_cast<uint8_t*>(data);
    uint32_t bytes_left = data_length;

    while (bytes_left > 0) {
        uint32_t chunk_size = std::min(kMaxBulkBufferSize, bytes_left);
        memcpy(BulkTransferBuffer, current_chunk, chunk_size);
        if (BulkOutTransferInternal(kSingleBulkOutEndpoint, BulkTransferBuffer, chunk_size)) {
            current_chunk += chunk_size;
            bytes_left -= chunk_size;
        } else {
            printf("Bad BulkOutTransferInternal\r\n");
            return false;
        }
    }

    return true;
}

bool TpuDriver::BulkInTransferInternal(uint8_t endpoint, uint8_t *data, uint32_t data_length) const {
    bool ret = false;
    SemaphoreHandle_t sema = xSemaphoreCreateBinary();

    usb_status_t bulk_status = USB_HostEdgeTpuBulkInRecv(usb_instance_, endpoint,
            data, data_length,
            [](void *param,
               uint8_t *data,
               uint32_t data_length,
               usb_status_t status) {
                SemaphoreHandle_t sema = (SemaphoreHandle_t)param;
                xSemaphoreGive(sema);
            }, sema);

    if (bulk_status != kStatus_USB_Success) {
        printf("USB_HostEdgeTpuBulkInRecv failed\r\n");
        goto exit;
    }

    if (xSemaphoreTake(sema, pdMS_TO_TICKS(200)) == pdFALSE) {
        printf("%s didn't get semaphore\r\n", __func__);
        goto exit;
    };

    ret = true;
exit:
    vSemaphoreDelete(sema);
    return ret;
}

bool TpuDriver::BulkInTransfer(uint8_t *data, uint32_t data_length) const {
    uint8_t *current_chunk = data;
    uint32_t bytes_left = data_length;
    while (bytes_left > 0) {
        uint32_t chunk_size = std::min(kMaxBulkBufferSize, bytes_left);
        if (BulkInTransferInternal(kSingleBulkOutEndpoint, BulkTransferBuffer, chunk_size)) {
            memcpy(current_chunk, BulkTransferBuffer, chunk_size);
            current_chunk += chunk_size;
            bytes_left -= chunk_size;
        } else {
            printf("Bad BulkInTransferInternal\r\n");
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> TpuDriver::PrepareHeader(DescriptorTag tag, uint32_t length) const {
    constexpr size_t kPacketHeaderRawDataSizeInBytes = 8;
    constexpr size_t kLengthSizeInBytes = sizeof(length);
    std::vector<uint8_t> header_packet(kPacketHeaderRawDataSizeInBytes);
    std::fill(header_packet.begin(), header_packet.end(), 0);
    memcpy(header_packet.data(), &length, kLengthSizeInBytes);

    *(header_packet.data() + sizeof(kLengthSizeInBytes)) =
        (static_cast<uint8_t>(tag) & 0xF);

    return header_packet;
}

bool TpuDriver::WriteHeader(DescriptorTag tag, uint32_t length) const {
    std::vector<uint8_t> header_packet = PrepareHeader(tag, length);
    return BulkOutTransfer(header_packet.data(), header_packet.size());
}

bool TpuDriver::ReadEvent() const {
    bool ret = false;
    constexpr size_t kEventSizeBytes = 16;
    uint8_t *buf = (uint8_t*)OSA_MemoryAllocate(kEventSizeBytes);
    SemaphoreHandle_t sema = xSemaphoreCreateBinary();
    usb_status_t bulk_status = USB_HostEdgeTpuBulkInRecv(usb_instance_,
            kEventInEndpoint,
            buf,
            kEventSizeBytes,
            [](void *param, uint8_t *data, uint32_t data_length, usb_status_t status) {
            uint32_t len;
            uint64_t address;
            uint8_t tag;
            memcpy(&address, data, sizeof(address));
            memcpy(&len, data + sizeof(address), sizeof(len));
            tag = *(data + sizeof(address) + sizeof(len)) & 0xF;
            // For now, we don't do anything with these events we've read back.
            (void)tag;
        SemaphoreHandle_t sema = (SemaphoreHandle_t)param;
        xSemaphoreGive(sema);
    }, sema);
    if (bulk_status != kStatus_USB_Success) {
        printf("ReadEvent failed\r\n");
        goto exit;
    }
    if (xSemaphoreTake(sema, pdMS_TO_TICKS(200)) == pdFALSE) {
        goto exit;
    };
    ret = true;
exit:
    vSemaphoreDelete(sema);
    OSA_MemoryFree(buf);
    return ret;
}

bool TpuDriver::DoRunControl(platforms::darwinn::driver::RunControl run_state) {
    const uint64_t run_state_value = static_cast<uint64_t>(run_state);
    assert(Write64(chip_config_.GetScalarCoreCsrOffsets().scalarCoreRunControl, run_state_value));
    assert(Write64(chip_config_.GetScalarCoreCsrOffsets().avDataPopRunControl, run_state_value));
    assert(Write64(chip_config_.GetScalarCoreCsrOffsets().parameterPopRunControl, run_state_value));
    assert(Write64(chip_config_.GetScalarCoreCsrOffsets().infeedRunControl, run_state_value));
    assert(Write64(chip_config_.GetScalarCoreCsrOffsets().outfeedRunControl, run_state_value));

    registers::TileConfig<7> helper;
    helper.set_broadcast();
    assert(Write64(chip_config_.GetTileConfigCsrOffsets().tileconfig0, helper.raw()));

    // Wait until tileconfig0 is set correctly. Subsequent writes are going to
    // tiles, but hardware does not guarantee correct ordering with previous
    // write.
    uint64_t tileconfig0_reg;
    do {
        assert(Read64(chip_config_.GetTileConfigCsrOffsets().tileconfig0, &tileconfig0_reg));
    } while (tileconfig0_reg != helper.raw());

    if (chip_config_.GetTileCsrOffsets().opRunControl != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().opRunControl, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().opRunControl_0 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().opRunControl_0, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().opRunControl_1 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().opRunControl_1, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().opRunControl_2 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().opRunControl_2, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().opRunControl_3 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().opRunControl_3, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().opRunControl_4 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().opRunControl_4, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().opRunControl_5 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().opRunControl_5, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().opRunControl_6 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().opRunControl_6, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().opRunControl_7 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().opRunControl_7, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().narrowToWideRunControl != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().narrowToWideRunControl,run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().narrowToWideRunControl_0 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().narrowToWideRunControl_0, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().narrowToWideRunControl_1 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().narrowToWideRunControl_1, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().narrowToWideRunControl_2 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().narrowToWideRunControl_2, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().narrowToWideRunControl_3 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().narrowToWideRunControl_3, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().narrowToWideRunControl_4 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().narrowToWideRunControl_4, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().narrowToWideRunControl_5 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().narrowToWideRunControl_5, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().narrowToWideRunControl_6 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().narrowToWideRunControl_6, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().narrowToWideRunControl_7 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().narrowToWideRunControl_7, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().wideToNarrowRunControl != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().wideToNarrowRunControl,run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_0 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_0, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_1 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_1, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_2 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_2, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_3 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_3, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_4 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_4, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_5 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_5, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_6 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_6, run_state_value));
    }
    if (chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_7 != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().wideToNarrowRunControl_7, run_state_value));
    }

    assert(Write64(chip_config_.GetTileCsrOffsets().meshBus0RunControl, run_state_value));
    assert(Write64(chip_config_.GetTileCsrOffsets().meshBus1RunControl, run_state_value));
    assert(Write64(chip_config_.GetTileCsrOffsets().meshBus2RunControl, run_state_value));
    assert(Write64(chip_config_.GetTileCsrOffsets().meshBus3RunControl, run_state_value));
    assert(Write64(chip_config_.GetTileCsrOffsets().ringBusConsumer0RunControl, run_state_value));
    assert(Write64(chip_config_.GetTileCsrOffsets().ringBusConsumer1RunControl, run_state_value));
    assert(Write64(chip_config_.GetTileCsrOffsets().ringBusProducerRunControl, run_state_value));
    if (chip_config_.GetTileCsrOffsets().narrowToNarrowRunControl != static_cast<uint64_t>(-1)) {
        assert(Write64(chip_config_.GetTileCsrOffsets().narrowToNarrowRunControl, run_state_value));
    }

    return true;
}

}  // namespace valiant