"""Independent transmitter helpers for synthetic QAM acceptance tests.

Extracted unchanged from the project transmitter ris_codebook_gui.py.
Source SHA-256: 3a25ab8b5b7f3ad64ef66f3d0bd6625de00f25397e44f2f37f199a5993d32991
No receiver decision code or payload truth is used by these helpers.
"""
import cmath
import math
from dataclasses import dataclass, field, fields

TAU = 2 * math.pi

@dataclass
class Config:
    rows: int = 20
    cols: int = 20
    row_pitch_mm: float = 2.54
    col_pitch_mm: float = 19.14
    carrier_ghz: float = 5.8
    tx_x_m: float = 0.0
    tx_y_m: float = 0.0
    tx_z_m: float = -0.4
    rx_mode: str = "near"  # near: spherical focusing, far: outgoing plane wave
    rx_range_m: float = 0.6
    scan_angles_deg: list[float] = field(default_factory=lambda: [0.0])
    scan_azimuth_deg: float = 0.0
    pwm_mhz: float = 1.0
    slots: int = 256
    repeats: int = 3
    harmonic: int = 1
    gamma0_amplitude: float = 1.0
    gamma0_phase_deg: float = 0.0
    gamma1_amplitude: float = 1.0
    gamma1_phase_deg: float = 180.0
    amplitude_scale: float = 1.0
    min_high_ns: float = 0.0
    min_low_ns: float = 0.0
    qam_order: int = 256

    @property
    def bits_per_symbol(self):
        return self.qam_order.bit_length() - 1

    @property
    def pam_peak(self):
        return math.isqrt(self.qam_order) - 1

    @property
    def units(self):
        return self.rows * self.cols

    @property
    def slot_ns(self):
        return 1000.0 / self.pwm_mhz / self.slots

    @property
    def delta(self):
        return (self.gamma1_amplitude * cmath.exp(1j * math.radians(self.gamma1_phase_deg))
                - self.gamma0_amplitude * cmath.exp(1j * math.radians(self.gamma0_phase_deg)))

    def validate(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in ("rows", "cols", "slots", "repeats", "harmonic", "qam_order"):
                if isinstance(v, bool) or not isinstance(v, int):
                    raise ValueError(f"{f.name} 必须是整数")
            elif f.name not in ("rx_mode", "scan_angles_deg"):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError(f"{f.name} 必须是有限数值")
        if self.qam_order not in (64, 256):
            raise ValueError("QAM 阶数只支持 64 或 256")
        if not 1 <= self.rows <= 128 or not 1 <= self.cols <= 128:
            raise ValueError("行数和列数须为 1～128")
        if not 16 <= self.slots <= 16384:
            raise ValueError("每 PWM 周期的时间片数须为 16～16384")
        if not 1 <= self.repeats <= 10000:
            raise ValueError("每符号重复次数须为 1～10000")
        if self.harmonic not in (-1, 1):
            raise ValueError("只支持 +1 或 -1 阶谐波")
        if min(self.row_pitch_mm, self.col_pitch_mm, self.carrier_ghz, self.pwm_mhz) <= 0:
            raise ValueError("间距、载频和 PWM 频率必须大于 0")
        if self.tx_z_m >= 0:
            raise ValueError("本透射模型要求发射端 z < 0，接收端位于 z > 0 一侧")
        if self.rx_mode not in ("near", "far"):
            raise ValueError("接收模型须为 near 或 far")
        if self.rx_range_m <= 0:
            raise ValueError("接收距离必须大于 0")
        if not isinstance(self.scan_angles_deg, list) or not 1 <= len(self.scan_angles_deg) <= 16:
            raise ValueError("目标角度需包含 1～16 个值")
        if any(isinstance(a, bool) or not isinstance(a, (int, float))
               or not math.isfinite(a) or abs(a) >= 90 for a in self.scan_angles_deg):
            raise ValueError("各扫描角必须在 -90° 与 +90° 之间")
        if self.gamma0_amplitude < 0 or self.gamma1_amplitude < 0 or abs(self.delta) < 1e-10:
            raise ValueError("两状态幅度不能为负，且复透射系数必须不同")
        if not 0 < self.amplitude_scale <= 1:
            raise ValueError("星座幅度系数须在 (0, 1] 范围")
        if min(self.min_high_ns, self.min_low_ns) < 0:
            raise ValueError("最短高/低电平时间不能为负；未知时填 0")
        if self.carrier_ghz * 1e9 + self.harmonic * self.pwm_mhz * 1e6 <= 0:
            raise ValueError("目标谐波频率必须为正")
        if self.p_limits()[0] > self.p_limits()[1]:
            raise ValueError("高/低电平最短时间约束不允许有效脉冲，请调整时间片数或 PWM 频率")
        combinations = len(self.scan_angles_deg) * self.qam_order * self.units
        if combinations > 8_000_000:
            raise ValueError("组合超过 800 万，请减少单元数或目标角度数")

    def p_limits(self):
        lo = max(1, math.ceil(self.min_high_ns / self.slot_ns - 1e-12))
        hi = min(self.slots // 2,
                 self.slots - max(1, math.ceil(self.min_low_ns / self.slot_ns - 1e-12)))
        return lo, hi


def gray_decode(g):
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n


def qam_symbol(symbol_id, qam_order=256):
    """Gray labels: legacy byte IIIIQQQQ (256), or six bits IIIQQQ (64)."""
    if qam_order == 64:
        if not 0 <= symbol_id < 64:
            raise ValueError("64QAM 符号须为 0～63")
        i = 2 * gray_decode(symbol_id >> 3) - 7
        q = 2 * gray_decode(symbol_id & 7) - 7
        return i, q, complex(i, q) / (7 * math.sqrt(2))
    if qam_order != 256:
        raise ValueError("QAM 阶数只支持 64 或 256")
    i = 2 * gray_decode(symbol_id >> 4) - 15
    q = 2 * gray_decode(symbol_id & 15) - 15
    return i, q, complex(i, q) / (15 * math.sqrt(2))


def coefficient(delta, p, r, slots, harmonic=1):
    """Exact integral over zero-order-held states on all slots (not a raw DFT)."""
    return (delta * math.sin(math.pi * p / slots) / math.pi
            * cmath.exp(-1j * harmonic * TAU * (r + p / 2) / slots))


def quantize_pulse(rho, phase, cfg):
    """Minimize |a_actual-a_target| over allowed widths and pulse locations.

    Integer widths have two center-parity classes. Within each class phase
    error is identical, so widths nearest the analytic optimum plus the
    endpoints give the global optimum without an O(L^2) search.
    """
    slots = cfg.slots
    lo, hi = cfg.p_limits()
    target = abs(cfg.delta) / math.pi * rho * cmath.exp(1j * phase)
    center = ((cmath.phase(cfg.delta) - phase) / (TAU * cfg.harmonic)) % 1.0
    best = None
    for parity in (0, 1):
        low = lo + (parity - lo) % 2
        high = hi - (hi - parity) % 2
        if low > high:
            continue
        r0 = math.floor(center * slots - parity / 2)
        # Residual phase magnitude is independent of p within a parity class.
        error_angle = TAU * ((r0 + parity / 2) / slots - center)
        error_angle = (error_angle + math.pi) % TAU - math.pi
        alternative = (error_angle + TAU / slots + math.pi) % TAU - math.pi
        error_angle = min((error_angle, alternative), key=abs)
        optimum = slots * math.asin(max(0.0, min(1.0, rho * math.cos(error_angle)))) / math.pi
        nearest = low + 2 * math.floor((optimum - low) / 2)
        widths = {low, high, max(low, min(high, nearest)),
                  max(low, min(high, nearest + 2))}
        for p in sorted(widths):
            start = center * slots - p / 2
            for r_unwrapped in (math.floor(start), math.floor(start) + 1):
                r = r_unwrapped % slots
                a = coefficient(cfg.delta, p, r, slots, cfg.harmonic)
                error = abs(a - target) ** 2
                if best is None or error < best[0]:
                    best = (error, p, r, a)
    return best[1], best[2], best[3], best[0]
