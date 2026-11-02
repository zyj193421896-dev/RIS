#ifndef RFMC6000_AD_CONFIG_H
#define RFMC6000_AD_CONFIG_H

#include "../drivers/ad9361.h"
#include "rf_profile.h"

/* Execute the user-exported GUI script instead of the reference no-OS init. */
#define AD_CONFIG_USE_GUI_SCRIPT 1
/* Compatibility name for the previous application. Profile selection is
 * now independent of choosing the GUI-script versus no-OS initialization. */
#define AD_CONFIG_USE_CONFIG915 AD_CONFIG_USE_GUI_SCRIPT

/* RFMC6000 example final RF profile, adapted to FMCOMMS3 crystal.
 * Optional RX-port BIST: Fs/16=3.84 MHz, -6 dBFS (0x3F4=0x5B). */
#define AD_CONFIG_SAMPLE_HZ 61440000U
#define AD_CONFIG_BANDWIDTH_HZ 30720000U
#define AD_CONFIG_BIST_PERIOD 16U
#define AD_CONFIG_BIST_TONE_HZ (AD_CONFIG_SAMPLE_HZ / AD_CONFIG_BIST_PERIOD)

enum ad_config_status {
    AD_CONFIG_OK = 0,
    AD_CONFIG_ERR_NOT_READY = -200,
    AD_CONFIG_ERR_RETRY = -201,
    AD_CONFIG_ERR_READBACK = -202,
    AD_CONFIG_ERR_PLL = -203
};

typedef struct {
    void *phy; /* Private reference-driver handle, not the SPI-only device. */
    u8 ready;
    u8 attempted;
} ad_config_device;

/* All fields are 32-bit for direct debugger inspection in PS DDR. */
typedef struct {
    u32 magic;
    u32 done;
    s32 result;
    u32 stage;
    s32 io_error;
    u32 spi_reads;
    u32 spi_writes;
    u32 last_reg;
    u32 last_value;
    u32 wait_reg;
    u32 wait_mask;
    s32 wait_value;
    u32 bbpll_status;
    u32 rxpll_status;
    u32 txpll_status;
    u32 calibration;
    u32 ensm_state;
    u32 port_conf[3];
    u32 rx_sample_hz;
    u32 tx_sample_hz;
    u32 rx_bandwidth_hz;
    u32 tx_bandwidth_hz;
    u32 rx_lo_mhz;
    u32 tx_lo_mhz;
    s32 rx_gain_db[2];
    u32 tx_attenuation_mdb[2];
    u32 rx_fir_enabled;
    u32 tx_fir_enabled;
    u32 elapsed_ms;
    u32 clock_enable; /* REG_CLOCK_ENABLE; XO_BYPASS must be clear for Y101. */
} ad_config_report_type;

extern volatile ad_config_report_type ad_config_report;

/* One attempt per PS/program reset. Reuse the initialized SPI0/SS0 and
 * RESETB GPIO 54 (FMCOMMS3 H31 / FPGA AC29). Apply the reference RF profile:
 * REF=40 MHz, Fs=61.44 MSPS, BW=30.72 MHz, RX LO=200 MHz, TX LO=5997 MHz,
 * MGC=5 dB, TX attenuation=0 dB. Capture ARM and DMA remain stopped.
 * FMCOMMS3 Y101 40 MHz crystal uses internal DCXO; no clock-chip dependency.
 * Negative errno values from the reference driver are propagated unchanged.
 */
int ad_config_init(ad_config_device *dev, ad9361_device *transport);
int ad_config_init_915(ad_config_device *dev, ad9361_device *transport);
int ad_config_init_2p4(ad_config_device *dev, ad9361_device *transport);
int ad_config_init_5p8(ad_config_device *dev, ad9361_device *transport);
int ad_config_init_selected(ad_config_device *dev, ad9361_device *transport);
/* Internal RX-port test tone at Fs/16=3.84 MHz, -6 dBFS; no RF input required.
 * enabled=0 restores ordinary RX samples. Does not arm PL/DMA or reset RF.
 */
int ad_config_rx_bist_tone(ad_config_device *dev, u8 enabled);
const char *ad_config_status_string(int status);

#endif
