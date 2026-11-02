#include "capture_store.h"
#include "../ad_config/rf_profile.h"
#include <stdio.h>
#include <string.h>
#include "xil_printf.h"
capture_store_record capture_store;
static dma_9361_device *dma;
static gpio_9361_device *gpio;
static int (*rearm_capture)(void);
static int (*read_configuration)(u8 *);
static int preparing;

void capture_store_init(dma_9361_device *d, gpio_9361_device *g,
                        int (*rearm)(void), int (*read_rf)(u8 *))
{
    dma = d; gpio = g; rearm_capture = rearm; read_configuration = read_rf;
    memset(&capture_store, 0, sizeof(capture_store));
    capture_store.state = CAP_STORE_EMPTY;
    capture_store.reset_verified = 1;
}
static void finish(int result)
{
    XTime now, debug_start;
    int reset;
    preparing = 1;
    XTime_GetTime(&now);
    capture_store.elapsed_us = (u32)((now - capture_store.start) * 1000000U / COUNTS_PER_SECOND);
    capture_store.completed_packets = dma->completed_packets;
    capture_store.received_bytes = dma->received_bytes;
    capture_store.dma_status = dma->last_dma_status;
    capture_store.bd_status = dma->last_bd_status;
    capture_store.capture_status = dma->last_capture_status;
    (void)gpio_9361_get_status(gpio, &capture_store.tail_status);
    if (!result && (capture_store.received_bytes != capture_store.layout.bytes ||
        capture_store.completed_packets != capture_store.layout.packets))
        result = DMA_9361_ERR_PACKET;
    if (!result && (read_configuration(capture_store.rf_after) ||
        memcmp(capture_store.rf_before, capture_store.rf_after, CAPTURE_RF_REGS)))
        result = DMA_9361_ERR_CONFIG;
    capture_store.dma_debug = *dma_9361_last_debug();
    XTime_GetTime(&debug_start);
    if (!result) {
        /* Tail descriptors have completed; source/FIFOs may overflow while
         * diagnosis runs, but DMA owns no further payload buffers. No UART
         * printing or new capture is allowed until stop/rearm is complete. */
        capture_debug_begin(&capture_store.debug, CAPTURE_DATA_BASE, capture_store.layout.bytes);
        if (capture_debug_measure(&capture_store.debug, 0U)) result = DMA_9361_ERR_MEMORY;
        if (!result) result = dma_9361_quiesce(dma);
        if (!result && capture_debug_measure(&capture_store.debug, 1U))
            result = DMA_9361_ERR_MEMORY;
    }
    reset = rearm_capture();
    capture_store.reset_verified = reset == 0;
    if (reset) result = DMA_9361_ERR_RESET_REQUIRED;
    if (!result && capture_debug_measure(&capture_store.debug, 2U))
        result = DMA_9361_ERR_MEMORY;
    XTime_GetTime(&now);
    capture_store.debug_elapsed_us = (u32)((now-debug_start)*1000000U/COUNTS_PER_SECOND);
    capture_store.result = result;
    capture_store.state = result ? CAP_STORE_FAILED : CAP_STORE_READY;
    capture_store_debug_print();
    preparing = 0;
}
int capture_store_begin(u32 bytes)
{
    capture_layout layout;
    XTime now;
    int result;
    if (!dma || capture_store.state == CAP_STORE_UNAVAILABLE) return DMA_9361_ERR_NOT_READY;
    if (capture_store.state == CAP_STORE_CAPTURING ||
        capture_store.state == CAP_STORE_READY) return DMA_9361_ERR_STATE;
    if (!capture_store.reset_verified) return DMA_9361_ERR_NOT_READY;
    if (capture_make_layout(bytes, &layout)) return DMA_9361_ERR_ARGUMENT;
    memset(&capture_store, 0, sizeof(capture_store));
    capture_store.first_read_bad_offset = 0xFFFFFFFFU;
    capture_store.layout = layout;
    capture_store.state = CAP_STORE_CAPTURING;
    capture_store.result = DMA_9361_PENDING;
    preparing = 1;
    XTime_GetTime(&now);
    snprintf(capture_store.id, sizeof(capture_store.id), "%08x%08x",
             (unsigned)(now >> 32), (unsigned)now);
    capture_store.start = now;
    result = read_configuration(capture_store.rf_before);
    if (!result) result = dma_9361_prepare(dma, CAPTURE_BD_BASE,
                                           CAPTURE_DATA_BASE, layout.packets);
    if (!result) result = dma_9361_start(dma);
    XTime_GetTime(&capture_store.start); /* acquisition timeout excludes prepare */
    preparing = 0;
    if (result) finish(result);
    return result;
}
void capture_store_poll(void)
{
    XTime now;
    int result;
    if (preparing || capture_store.state != CAP_STORE_CAPTURING) return;
    result = dma_9361_poll(dma);
    capture_store.completed_packets = dma->completed_packets;
    capture_store.received_bytes = dma->received_bytes;
    if (result != DMA_9361_PENDING) { finish(result); return; }
    XTime_GetTime(&now);
    if (now - capture_store.start >= (XTime)capture_store.layout.timeout_us *
                                    COUNTS_PER_SECOND / 1000000U)
        finish(DMA_9361_ERR_TIMEOUT);
}
void capture_store_abort(void)
{
    if (capture_store.state == CAP_STORE_CAPTURING && !preparing)
        finish(DMA_9361_ERR_ABORTED);
}
int capture_store_release(const char *id)
{
    if (capture_store.state != CAP_STORE_READY || strcmp(capture_store.id, id)) return -1;
    capture_store.state = CAP_STORE_EMPTY;
    return 0;
}
int capture_store_read(const char *id, u32 offset, u32 bytes, void *dst)
{
    u32 i;
    const unsigned char *data = dst;
    if (!id || !dst || capture_store.state != CAP_STORE_READY || strcmp(capture_store.id, id) ||
        !bytes || bytes > CAPTURE_CHUNK_BYTES || (offset | bytes) % CAPTURE_TUPLE_BYTES ||
        offset > capture_store.layout.bytes || bytes > capture_store.layout.bytes - offset)
        return -1;
    if (mig_ddr_read(&dma->memory, CAPTURE_DATA_BASE + offset, dst, bytes) != MIG_DDR_OK)
        return -1;
    ++capture_store.read_ops;
    /* Count exactly the bytes sent, without changing or suppressing invalid
     * IQ. Counts describe read requests, including retries, not unique samples. */
    for (i = 0U; i < bytes; i += 2U) {
        u32 value = (u32)data[i] | (u32)data[i+1U] << 8;
        if (value > 0x07FFU && value < 0xF800U) {
            ++capture_store.read_bad[(i/2U) & 3U];
            if (capture_store.first_read_bad_offset == 0xFFFFFFFFU) {
                capture_store.first_read_bad_offset = offset+i;
                capture_store.first_read_bad_raw = value;
            }
        }
    }
    return 0;
}
int capture_store_keys_blocked(void)
{ return capture_store.state == CAP_STORE_CAPTURING; }
int capture_store_urgent(void)
{
    return capture_store.state == CAP_STORE_CAPTURING && !preparing &&
        capture_store.layout.packets - capture_store.completed_packets <= 8U;
}
const char *capture_store_state_name(void)
{
    static const char *const names[] = {"unavailable", "empty", "capturing", "ready", "failed"};
    return names[capture_store.state];
}
void capture_store_debug_print(void)
{
    if (capture_store.state == CAP_STORE_CAPTURING) return;
    if (!capture_store.layout.bytes) {
        dma_diag_print(dma_9361_last_debug(), "last_transfer");
        xil_printf("RXDBG no bulk capture record; g/s diagnostics are separate\r\n");
        return;
    }
    xil_printf("RXDBG WINDOW id=%s state=%s result=%d requested=%u received=%u packets=%u/%u reset_verified=%d\r\n",
        capture_store.id, capture_store_state_name(), capture_store.result,
        capture_store.layout.bytes, capture_store.received_bytes,
        capture_store.completed_packets, capture_store.layout.packets, capture_store.reset_verified);
    dma_diag_print(&capture_store.dma_debug, capture_store.id);
    capture_debug_print(&capture_store.debug, capture_store.id);
    xil_printf("RXDBG READ requests=%u bad_I1Q1I2Q2=%u/%u/%u/%u first_offset=%08x raw=%04x diagnostic_us=%u\r\n",
        capture_store.read_ops, capture_store.read_bad[0], capture_store.read_bad[1],
        capture_store.read_bad[2], capture_store.read_bad[3], capture_store.first_read_bad_offset,
        capture_store.first_read_bad_raw, capture_store.debug_elapsed_us);
}
int capture_store_debug_json(char *buffer, unsigned capacity)
{
    if (capture_store.state == CAP_STORE_CAPTURING) return -1;
    return capture_debug_json(&capture_store.debug, capture_store.id, buffer, capacity);
}
int capture_store_json(char *b, unsigned capacity)
{
    return snprintf(b, capacity,
        "{\"ok\":true,\"protocol\":1,\"state\":\"%s\",\"capture_id\":\"%s\","
        "\"bytes\":%u,\"received_bytes\":%u,\"tuples\":%u,\"packets\":%u,\"completed_packets\":%u,"
        "\"sample_rate_hz\":%u,\"rx_lo_config_hz\":%llu,\"rf_profile\":\"%s\",\"channels\":2,"
        "\"order\":[\"I1\",\"Q1\",\"I2\",\"Q2\"],\"dtype\":\"int16\",\"byte_order\":\"little\","
        "\"valid_bits\":12,\"data_base\":%u,\"max_bytes\":%u,\"chunk_bytes\":%u,"
        "\"address_mode\":\"banked-gp0\",\"physical_ddr_bytes\":2147483648,\"cpu_window_bytes\":1073741824,"
        "\"duration_us\":%u,\"elapsed_us\":%u,\"result\":%d,\"reset_verified\":%d,"
        "\"dma_status\":%u,\"bd_status\":%u,\"capture_status\":%u,\"tail_status\":%u,"
        "\"debug_version\":1,\"debug_probe_bytes\":%u,\"debug_phases\":%u,"
        "\"debug_changed_dma\":%u,\"debug_changed_pl\":%u,\"debug_unstable\":%u,\"debug_memcpy_diff\":%u,"
        "\"debug_elapsed_us\":%u,\"debug_read_ops\":%u,\"debug_read_bad\":[%u,%u,%u,%u],"
        "\"debug_first_bad_offset\":%u,\"debug_first_bad_raw\":%u,"
        "\"debug_bd_issues\":%u,\"debug_bd_index\":%u,\"debug_bd_raw_status\":%u,\"debug_bd_actual\":%u,"
        "\"debug_bd_words\":[%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u],"
        "\"rf_regs\":[6,16,250,251,253,265,268,1012,1013,1014,3,4,688,693],"
        "\"rf_before\":[%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u],"
        "\"rf_after\":[%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u]}\n",
        capture_store_state_name(), capture_store.id,
        (unsigned)capture_store.layout.bytes, (unsigned)capture_store.received_bytes,
        (unsigned)capture_store.layout.tuples, (unsigned)capture_store.layout.packets,
        (unsigned)capture_store.completed_packets, (unsigned)CAPTURE_SAMPLE_RATE,
        (unsigned long long)RX_RF_LO_HZ, RX_RF_PROFILE_NAME,
        (unsigned)CAPTURE_DATA_BASE, (unsigned)CAPTURE_MAX_BYTES, (unsigned)CAPTURE_CHUNK_BYTES,
        (unsigned)capture_store.layout.duration_us, (unsigned)capture_store.elapsed_us,
        capture_store.result, capture_store.reset_verified,
        (unsigned)capture_store.dma_status,(unsigned)capture_store.bd_status,
        (unsigned)capture_store.capture_status,(unsigned)capture_store.tail_status,
        (unsigned)capture_store.debug.count*CAPTURE_DEBUG_PROBE_BYTES, (unsigned)capture_store.debug.phases,
        (unsigned)capture_store.debug.changed_dma, (unsigned)capture_store.debug.changed_pl,
        (unsigned)capture_store.debug.unstable, (unsigned)capture_store.debug.copy_mismatch,
        (unsigned)capture_store.debug_elapsed_us, (unsigned)capture_store.read_ops,
        (unsigned)capture_store.read_bad[0], (unsigned)capture_store.read_bad[1], (unsigned)capture_store.read_bad[2], (unsigned)capture_store.read_bad[3],
        (unsigned)capture_store.first_read_bad_offset, (unsigned)capture_store.first_read_bad_raw,
        (unsigned)capture_store.dma_debug.issues, (unsigned)capture_store.dma_debug.index,
        (unsigned)capture_store.dma_debug.raw[7], (unsigned)capture_store.dma_debug.raw[7] & (unsigned)capture_store.dma_debug.max_length,
        (unsigned)capture_store.dma_debug.raw[0], (unsigned)capture_store.dma_debug.raw[1], (unsigned)capture_store.dma_debug.raw[2], (unsigned)capture_store.dma_debug.raw[3],
        (unsigned)capture_store.dma_debug.raw[4], (unsigned)capture_store.dma_debug.raw[5], (unsigned)capture_store.dma_debug.raw[6], (unsigned)capture_store.dma_debug.raw[7],
        (unsigned)capture_store.dma_debug.raw[8], (unsigned)capture_store.dma_debug.raw[9], (unsigned)capture_store.dma_debug.raw[10], (unsigned)capture_store.dma_debug.raw[11],
        (unsigned)capture_store.dma_debug.raw[12], (unsigned)capture_store.dma_debug.raw[13], (unsigned)capture_store.dma_debug.raw[14], (unsigned)capture_store.dma_debug.raw[15],
        capture_store.rf_before[0],capture_store.rf_before[1],capture_store.rf_before[2],capture_store.rf_before[3],
        capture_store.rf_before[4],capture_store.rf_before[5],capture_store.rf_before[6],capture_store.rf_before[7],
        capture_store.rf_before[8],capture_store.rf_before[9],capture_store.rf_before[10],capture_store.rf_before[11],
        capture_store.rf_before[12],capture_store.rf_before[13],
        capture_store.rf_after[0],capture_store.rf_after[1],capture_store.rf_after[2],capture_store.rf_after[3],
        capture_store.rf_after[4],capture_store.rf_after[5],capture_store.rf_after[6],capture_store.rf_after[7],
        capture_store.rf_after[8],capture_store.rf_after[9],capture_store.rf_after[10],capture_store.rf_after[11],
        capture_store.rf_after[12],capture_store.rf_after[13]);
}
