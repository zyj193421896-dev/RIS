"""Bounded, read-only FFT analysis of little-endian I1 Q1 I2 Q2 captures."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


class AnalysisCancelled(Exception):
    pass


@dataclass
class Spectrum:
    path: Path
    channel: int
    sample_rate_hz: float
    rate_source: str
    lo_hz: float | None
    first_tuple: int
    total_tuples: int
    nfft: int
    averages: int
    remove_dc: bool
    frequency_hz: object
    dbfs: object
    peak_hz: float | None
    peak_dbfs: float | None
    rms_counts: float
    clipped: int
    power_dbfs: float
    ac_power_dbfs: float
    dc_iq: complex
    papr_db: float | None
    constellation: object
    constellation_indices: object
    block_time_s: object
    block_power_dbfs: object
    peaks: object
    occupied_99_hz: object
    waveform: object
    reference_waveform: object

    @property
    def samples(self):
        return self.nfft * self.averages


def _cancel(event):
    if event is not None and event.is_set():
        raise AnalysisCancelled("频谱分析已停止")


def _metadata(path, size, fallback_rate):
    sidecar = Path(str(path) + ".json")
    if not sidecar.exists():
        return fallback_rate, "手动设置（没有同名 JSON）", None
    try:
        if sidecar.stat().st_size > 1024 * 1024:
            raise ValueError("JSON 超过 1 MiB")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        capture = meta["capture"]
        if (meta.get("complete") is not True or meta["file_bytes"] != size
                or capture["bytes"] != size or capture["tuples"] != size // 8
                or capture["order"] != ["I1", "Q1", "I2", "Q2"]
                or capture["dtype"] != "int16" or capture["byte_order"] != "little"
                or capture["channels"] != 2 or capture["valid_bits"] != 12):
            raise ValueError("文件长度、完成标记或 IQ 格式不匹配")
        rate = float(capture["sample_rate_hz"])
        lo = capture.get("rx_lo_config_hz")
        if lo is not None:
            lo = float(lo)
            if not math.isfinite(lo) or lo <= 0:
                raise ValueError("无效的 LO 配置值")
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("无效的采样率")
        return rate, "同名 JSON", lo
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"同名 JSON 无效：{exc}") from exc


def analyze_bin(filename, *, channel=1, nfft=65536, averages=16, first_tuple=0,
                sample_rate_hz=15360000, remove_dc=True, cancel_event=None):
    """Average nonoverlapping Hann-windowed complex FFTs in linear power.

    Only one FFT block is read at a time. For a bin-centred complex sinusoid
    with I/Q amplitude 2048 counts, the peak is 0 dBFS. This is a coherent-gain
    corrected bin spectrum, not a PSD or calibrated RF power measurement.
    Metadata supplies Fs when available; otherwise the explicit fallback wins.
    """
    _cancel(cancel_event)
    import numpy as np

    path = Path(filename).expanduser()
    if path.name.endswith(".part"):
        raise ValueError("请选择下载完成的 bin 文件，不能分析 .part")
    if channel not in (1, 2):
        raise ValueError("通道必须是 RX1 或 RX2")
    if type(nfft) is not int or not 256 <= nfft <= 1048576 or nfft & (nfft - 1):
        raise ValueError("FFT 点数必须是 256～1048576 之间的 2 次幂")
    if type(averages) is not int or not 1 <= averages <= 64:
        raise ValueError("平均段数必须是 1～64")
    if type(first_tuple) is not int or first_tuple < 0:
        raise ValueError("起始点必须是非负整数（每通道采样点）")
    sample_rate_hz = float(sample_rate_hz)
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("采样率必须是正的有限数值")
    # Open only once: the size and samples refer to the same file handle.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if not size or size % 8:
            raise ValueError("bin 长度必须是 8 字节的正整数倍（I1/Q1/I2/Q2）")
        rate, source, lo = _metadata(path, size, sample_rate_hz)
        available = size // 8 - first_tuple
        if available < 256:
            raise ValueError("起始位置之后至少需要 256 个 IQ 采样点")
        nfft = min(nfft, 1 << (available.bit_length() - 1))
        averages = min(averages, available // nfft)
        stream.seek(first_tuple * 8)
        window = np.hanning(nfft + 1)[:-1]  # periodic Hann, coherent gain 0.5
        scale = float(window.sum()) ** 2
        power = np.zeros(nfft, dtype=np.float64)
        energy, clipped = 0.0, 0
        ac_energy, dc_sum, max_power = 0.0, 0j, 0.0
        points, point_indices, block_powers = [], [], []
        point_budget = max(1, 8192 // averages)
        for block_index in range(averages):
            _cancel(cancel_event)
            raw = np.fromfile(stream, dtype="<i2", count=nfft * 4)
            if raw.size != nfft * 4:
                raise ValueError("文件在分析期间被截短")
            iq = raw.reshape(-1, 4)[:, (channel - 1) * 2:channel * 2]
            invalid = (iq < -2048) | (iq > 2047)
            if np.any(invalid):
                row, component = divmod(int(np.argmax(invalid)), 2)
                sample = first_tuple + block_index * nfft + row
                offset = sample * 8 + (channel - 1) * 4 + component * 2
                value = int(iq[row, component])
                label = f"{'IQ'[component]}{channel}"
                raise ValueError(
                    f"RX{channel} {label} 采样点 {sample}（从 0 计），"
                    f"文件偏移 {offset}（0x{offset:X}）："
                    f"数值 {value}（0x{value & 0xffff:04X}）"
                    "不是有效的 12 位符号扩展。已停止分析；"
                    "请检查采集及存储/读出数据的完整性。")
            clipped += int(np.count_nonzero(np.any((iq <= -2048) | (iq >= 2047), axis=1)))
            x = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
            if block_index == 0:
                waveform = x[:4096].copy()
                other = raw.reshape(-1, 4)[:len(waveform), (2 if channel == 1 else 0): (4 if channel == 1 else 2)]
                reference_waveform = other[:, 0].astype(float) + 1j*other[:, 1].astype(float)
                bad_reference = np.any((other < -2048) | (other > 2047), axis=1)
                reference_waveform[bad_reference] = complex(float('nan'), float('nan'))
            energy += float(np.vdot(x, x).real)
            dc_sum += complex(x.sum())
            ac_energy += float(np.vdot(x - x.mean(), x - x.mean()).real)
            max_power = max(max_power, float(np.max(abs(x) ** 2)))
            block_powers.append(float(np.mean(abs(x) ** 2)))
            # Contiguous short runs avoid a regular decimator hiding tone phase.
            keep = min(point_budget, nfft)
            points.append(x[:keep].copy())
            point_indices.append(first_tuple + block_index * nfft + np.arange(keep))
            if remove_dc:
                x -= x.mean()
            transformed = np.fft.fft((x / 2048.0) * window)
            power += (transformed.real ** 2 + transformed.imag ** 2) / scale
        _cancel(cancel_event)
        stream.seek(0, 2)
        if stream.tell() != size:
            raise ValueError("文件长度在分析期间发生变化，请使用下载完成的文件")
    power = np.fft.fftshift(power / averages)
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / rate))
    dbfs = 10.0 * np.log10(np.maximum(power, 1e-18))
    peak = int(np.argmax(power))
    has_peak = power[peak] > 1e-18
    count = nfft * averages
    def level(value):
        return float(10 * np.log10(max(value / 2048.0 ** 2, 1e-18)))
    candidates = np.flatnonzero((power >= np.roll(power, 1)) &
                               (power > np.roll(power, -1)) & (power > 1e-18))
    strongest = []
    for index in candidates[np.argsort(power[candidates])[::-1]]:
        if all(min(abs(int(index)-k), nfft-abs(int(index)-k)) >= 4 for k in strongest):
            strongest.append(int(index))
        if len(strongest) == 5:
            break
    peaks = [(float(frequencies[k]), float(dbfs[k])) for k in strongest]
    occupied = None
    if power.sum() > 1e-18:
        cumulative = np.cumsum(power) / power.sum()
        low, high = np.searchsorted(cumulative, [.005, .995])
        occupied = (float(frequencies[low]), float(frequencies[min(high, nfft-1)]))
    return Spectrum(path, channel, rate, source, lo, first_tuple, size // 8, nfft, averages,
                    remove_dc, frequencies, dbfs,
                    float(frequencies[peak]) if has_peak else None,
                    float(dbfs[peak]) if has_peak else None,
                    math.sqrt(energy / count), clipped,
                    level(energy / count), level(ac_energy / count), dc_sum / count,
                    float(10*np.log10(max_power/(energy/count))) if energy else None,
                    np.concatenate(points), np.concatenate(point_indices),
                    (first_tuple + (np.arange(averages)+.5)*nfft)/rate,
                    np.array([level(v) for v in block_powers]), peaks, occupied,
                    waveform, reference_waveform)


def phase_trace(result, *, shifted=False, floor_dbfs=-45.0):
    """Contiguous waveform phases; never unwrap decimated/gapped constellation points.

    The floor is a display amplitude gate, not an estimate of signal-to-noise ratio.
    Relative phase uses simultaneous raw channels and requires a coherent reference
    for a propagation/channel interpretation. It is not a reflection-phase estimate.
    """
    import numpy as np
    if not math.isfinite(floor_dbfs):
        raise ValueError('相位显示门限必须是有限的 dBFS 数值')
    raw = result.waveform
    x = raw.copy()
    time = np.arange(len(x))/result.sample_rate_hz
    if shifted:
        x -= result.dc_iq
        if result.peak_hz is not None:
            x *= np.exp(-2j*np.pi*result.peak_hz*time)
    minimum = 2048.0 * 10**(floor_dbfs/20)
    valid = np.isfinite(x) & (abs(x) >= minimum) & (abs(x) > 0)
    phase = np.where(valid, np.angle(x, deg=True), np.nan)
    unwrapped = np.full(len(x), np.nan)
    # Restart at every low-amplitude gap; phase through such gaps is undefined.
    edges = np.diff(np.r_[False, valid, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        unwrapped[start:end] = np.rad2deg(np.unwrap(np.angle(x[start:end])))
    ref = result.reference_waveform
    relative_valid = (np.isfinite(ref) & (abs(ref) >= minimum) & (abs(ref) > 0) &
                      (abs(raw) >= minimum) & (abs(raw) > 0))
    relative = np.where(relative_valid, np.angle(raw*ref.conj(), deg=True), np.nan)
    return dict(time_s=time+result.first_tuple/result.sample_rate_hz,
                iq=x, wrapped_deg=phase, unwrapped_deg=unwrapped,
                relative_deg=relative, valid=valid)


def display_trace(result, max_points=8192):
    """Preserve the strongest bin in each display bucket; analysis stays full-size."""
    import numpy as np
    count = len(result.dbfs)
    width = max(1, (count + max_points - 1) // max_points)
    starts = np.arange(0, count, width)
    powers = np.maximum.reduceat(result.dbfs, starts)
    # Bucket centres keep the plotted frequency axis monotonic.
    centres = np.minimum(starts + width // 2, count - 1)
    return result.frequency_hz[centres] / 1e6, powers
