#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/nxp/rt1176-sdk/middleware/multicore/mcmgr/src/mcmgr.h"

#if __CORTEX_M == 7
void vGeneratePrimaryToSecondaryInterrupt(void* pxStreamBuffer) {
    MCMGR_TriggerEventForce(kMCMGR_FreeRtosMessageBuffersEvent, (uint32_t)pxStreamBuffer & 0xFFFF);
}
#else
void vGenerateSecondaryToPrimaryInterrupt(void* pxStreamBuffer) {
    MCMGR_TriggerEventForce(kMCMGR_FreeRtosMessageBuffersEvent, (uint32_t)pxStreamBuffer & 0xFFFF);
}
#endif
