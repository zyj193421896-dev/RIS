"""Strict GUI-independent adapter for checksum-bound 256QAM offline replay."""
from pathlib import Path
import json
from rx706_iq_replay import read_recipe, validate_recipe, verify_input
from rx706_spectrum import analyze_bin, _cancel
from rx706_qam import recover_qam


def validate_replay(recipe, input_path=None):
    def keys(value, allowed, label):
        if not isinstance(value, dict):
            raise ValueError(label + '须为参数对象')
        extra = set(value) - set(allowed.split())
        if extra:
            raise ValueError(label + '包含未知字段：' + ', '.join(sorted(extra)))
    keys(recipe, 'version method input_path input_fingerprint spectrum recovery caveat', '方案')
    keys(recipe.get('input_fingerprint'), 'file_bytes sha256', '指纹')
    keys(recipe.get('spectrum'), 'sample_rate_hz channel first_tuple nfft averages remove_dc', '频谱')
    r = recipe.get('recovery')
    keys(r, 'duration_s center_hint_hz order symbol_hz tdcm_options pwm_preamble_options preamble_options', '恢复')
    td = r.get('tdcm_options')
    keys(td, 'acquisition_mode base_hz harmonic normalization amplitude_scale phase_deg payload_symbols payload_repeat payload_data_symbols payload_tail_symbols crc_debug compensation', 'TDCM')
    keys(td.get('compensation'), 'enabled clock_mode period_ppm phase_mode frame_mode frame_symbols max_frames calibration_frames fit_timing timing_offset_symbols clock_search_ppm phase_update_hz phase_bandwidth_hz max_input_samples timing_search_symbols quality_evm_percent', '补偿')
    if 'crc_debug' in td:
        keys(td['crc_debug'], 'enabled', 'CRC')
    keys(r.get('pwm_preamble_options'), 'total_chips prefix_chips guard_chips symbols symbol_hz gap_s search_s max_error_percent width_offset_chips max_frequency_hz qam_search_symbols', 'PWM前导')
    keys(r.get('preamble_options'), 'symbols representation max_frequency_hz max_evm_percent search_s orientation fit_dc_offset', 'QAM训练')
    recipe = validate_recipe(recipe)
    if recipe['recovery']['order'] != 256:
        raise ValueError('256QAM功能区只接受256QAM绑定方案')
    if input_path is not None and Path(input_path).expanduser().resolve() != Path(recipe['input_path']).expanduser().resolve():
        raise ValueError('当前BIN与方案绑定的文件不同；请重新选择匹配方案')
    return recipe


def load_replay(path, input_path=None):
    """Validate all saved options and verify the complete bound IQ fingerprint."""
    path = Path(path)
    if path.stat().st_size > 512 * 1024:
        raise ValueError('方案JSON超过512KiB')
    recipe = validate_replay(json.loads(path.read_text(encoding='utf-8')), input_path)
    verify_input(recipe)
    return recipe


def run_replay(recipe, cancel_event=None):
    """Return (Spectrum, recovery); never discard options or silently change Fs."""
    recipe = validate_replay(recipe)
    _cancel(cancel_event)
    path = verify_input(recipe)
    spectrum = analyze_bin(path, **recipe['spectrum'], cancel_event=cancel_event)
    if spectrum.sample_rate_hz != recipe['spectrum']['sample_rate_hz']:
        raise ValueError('BIN元数据采样率与绑定方案不符；已停止复算')
    if spectrum.nfft != recipe['spectrum']['nfft'] or spectrum.averages != recipe['spectrum']['averages']:
        raise ValueError('BIN可用范围不足以保持绑定频谱参数；已停止复算')
    result = recover_qam(spectrum, **recipe['recovery'], cancel_event=cancel_event)
    _cancel(cancel_event)
    verify_input(recipe)
    return spectrum, result
