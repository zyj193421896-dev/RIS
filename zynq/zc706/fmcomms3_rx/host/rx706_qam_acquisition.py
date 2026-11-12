"""Automatic RF acquisition using known PWM headers and QAM training only.

Never uses payload labels, payload EVM, repeated payload, or a capture fingerprint
for selecting the carrier or clock. Legacy recover_qam remains unchanged.
"""
import copy
import math
import numpy as np
from rx706_spectrum import _cancel
from rx706_recovery import _read_span
from rx706_qam_pwm import acquire_pwm_header, validate_pwm_header, header_edges, smooth_pwm_observation
from rx706_qam_preamble import validate_preamble, _sample
from rx706_tdcm import harmonic_coefficients, _pwm_reference, validate_tdcm
from rx706_qam_training import fit_training, validate_orientation
from rx706_tdcm_debug import _header_fit, validate_compensation


def _carrier_candidates(x, fs, harmonic_hz, hint=None):
    """Spectral proposals, including both signs and +/- harmonic offsets."""
    n = min(1048576, 1 << (len(x).bit_length()-1))
    if n < 256:
        raise ValueError('自动载波搜索的采样长度不足')
    window = np.hanning(n+1)[:-1]
    power = np.zeros(n)
    for i in range(min(4, len(x)//n)):
        z = x[i*n:(i+1)*n]
        power += abs(np.fft.fft((z-z.mean())*window))**2
    freq = np.fft.fftfreq(n, 1/fs)
    local = (power > np.roll(power, 1)) & (power >= np.roll(power, -1))
    ranked = np.flatnonzero(local)
    ranked = ranked[np.argsort(power[ranked])[::-1]]
    peaks = []
    for at in ranked:
        f = float(freq[at])
        if all(abs(f-v) >= max(fs/n*3, abs(harmonic_hz)*.08) for v in peaks):
            peaks.append(f)
        if len(peaks) == 6:
            break
    candidates = []
    # The strongest FFT peak is only a proposal, never the acceptance criterion.
    for offset in (0., -harmonic_hz, harmonic_hz):
        for f in peaks:
            for sign in (1, -1):
                value = sign*(f+offset)
                if abs(value)+abs(harmonic_hz) < fs/2 and all(abs(value-v) > fs/n/2 for v in candidates):
                    candidates.append(float(value))
    if hint is not None and math.isfinite(hint) and abs(hint)+abs(harmonic_hz) < fs/2:
        if all(abs(hint-v) > fs/n/2 for v in candidates):
            candidates.append(float(hint))
    return candidates, peaks


def _known_header_clock(x, fs, pwm, anchor, nominal_frame, ndata, qam_rate,
                        bound_ppm, cancel_event=None, max_headers=20):
    """Fit a clock on the first known headers; the remaining frames are held out.

    The whole-frame spacing is a protocol constraint, but only the known PWM
    samples contribute to the fit. Payload may change arbitrarily every frame.
    """
    period = fs/pwm['symbol_hz']
    size = int(round(len(pwm['symbols'])*period))
    # Calibration is bounded to the first max_headers complete headers.
    first_k = int(math.ceil(-anchor/nominal_frame))
    available = int(math.floor((len(x)-size-2-anchor)/nominal_frame))
    ks = np.arange(first_k, min(available+1, first_k+max_headers), dtype=int)
    if len(ks) < 6:
        raise ValueError('自动前导时钟至少需要6个完整已知PWM前导；请增加恢复时长')
    end = min(len(x), int(anchor+ks[-1]*nominal_frame+size+3*period+2))
    source = smooth_pwm_observation(x[:end], fs, pwm)
    sums = np.r_[0j, np.cumsum(source)]
    powers = np.r_[0., np.cumsum(abs(source)**2)]
    rises, falls = header_edges(pwm, period)
    nh = int(np.sum(falls-rises)); nl = size-nh
    high_mask = np.zeros(size, bool)
    for a, b in zip(rises, falls):
        high_mask[a:b] = True
    local_positions = np.arange(size)
    measured = []
    for k in ks:
        _cancel(cancel_event)
        prediction = anchor+k*nominal_frame
        reach = max(period*.12, abs(k)*nominal_frame*bound_ppm*1e-6+period*.08)
        reach = min(reach, period*2)
        left = max(0, int(math.floor(prediction-reach)))
        right = min(len(source)-size-2, int(math.ceil(prediction+reach)))
        if right <= left:
            continue
        positions = np.arange(left, right+1, max(1, int(period/160)))
        high = np.zeros(len(positions), complex)
        for a, b in zip(rises, falls):
            high += sums[positions+b]-sums[positions+a]
        whole = sums[positions+size]-sums[positions]
        low = whole-high
        energy = powers[positions+size]-powers[positions]
        variance = np.maximum(0, energy-abs(high)**2/nh-abs(low)**2/nl)/size
        score = np.sqrt(variance)/np.maximum(abs(high/nh-low/nl), 1e-15)
        at = int(positions[np.argmin(score)])
        best = None
        for candidate in np.arange(max(0, at-5), min(len(source)-size-2, at+5)+.01, .25):
            z = _sample(source, candidate+local_positions)
            low_level, high_level = z[~high_mask].mean(), z[high_mask].mean()
            error = float(100*np.sqrt(np.mean(abs(z-np.where(high_mask,high_level,low_level))**2))/max(abs(high_level-low_level),1e-15))
            if best is None or error < best[0]:
                best = (error, float(candidate))
        if best is not None and best[0] <= pwm['max_error_percent']:
            measured.append((int(k), best[1], float(best[0])))
    if len(measured) < 6:
        raise ValueError('不足6个独立PWM前导通过匹配，不能自动估计时钟')
    measured = np.asarray(measured, float)
    good = np.ones(len(measured), bool)
    for _ in range(3):
        slope, intercept = np.polyfit(measured[good,0], measured[good,1], 1)
        residual = measured[:,1]-(intercept+slope*measured[:,0])
        median = np.median(residual[good])
        mad = np.median(abs(residual[good]-median))
        good = abs(residual-median) <= max(2.0, 4*1.4826*mad)
        if np.count_nonzero(good) < 6:
            raise ValueError('已知PWM前导时序不一致，拒绝自动时钟假锁')
    slope, intercept = np.polyfit(measured[good,0], measured[good,1], 1)
    residual = measured[good,1]-(intercept+slope*measured[good,0])
    nominal_cycles = len(pwm['symbols'])*period+ndata*fs/qam_rate
    scale = (slope-pwm['gap_s']*fs)/nominal_cycles
    ppm = float((scale-1)*1e6)
    if abs(ppm) > bound_ppm:
        raise ValueError('已知前导实测时钟超出配置的ppm搜索范围')
    if np.ptp(measured[good,0])*nominal_frame/fs < .025 or np.percentile(abs(residual),95) > max(3., period*.005):
        raise ValueError('已知前导时钟残差过大或跨度不足，拒绝外推')
    evidence = dict(period_ppm=ppm, frame_period_samples=float(slope),
        frame_period_s=float(slope/fs), measured_symbol_hz=float(qam_rate/scale),
        calibration_headers=int(np.count_nonzero(good)),
        calibration_end_s=float((measured[good,1].max()+size)/fs),
        header_indices=measured[good,1].tolist(),
        header_errors_percent=measured[good,2].tolist(),
        residual_samples=residual.tolist(), source='known_pwm_headers_only',
        payload_used=False)
    return ppm, evidence


def recover_auto_qam(spectrum, *, duration_s=1, center_hint_hz=None, symbol_hz=10000,
                     order=256, tdcm_options=None, pwm_preamble_options=None,
                     preamble_options=None, cancel_event=None):
    """Search RF origin, verify known training, fit known-header clock, decode.

    The input configuration is not mutated. The returned diagnostics contain
    the exact effective configuration and rejected RF candidates for replay.
    """
    fs = spectrum.sample_rate_hz
    td = copy.deepcopy(tdcm_options or {})
    cfg = validate_tdcm(fs, symbol_hz, td)
    comp = validate_compensation(td.get('compensation'))
    if not comp['enabled'] or cfg.get('acquisition_mode','pwm') != 'pwm':
        raise ValueError('自动载波/前导时钟仅支持启用补偿且含已知PWM前导的TDCM配置')
    if pwm_preamble_options is None or preamble_options is None:
        raise ValueError('自动搜索必须同时提供已知PWM前导与真实QAM训练，不能用未知净荷验收')
    pwm_preamble_options = copy.deepcopy(pwm_preamble_options)
    pwm_preamble_options.setdefault('observation_average_chips', 1.)
    pwm = validate_pwm_header(pwm_preamble_options, fs)
    points, training_bound, tolerance, _ = validate_preamble(preamble_options, order, symbol_hz)
    orientation, dc = validate_orientation(preamble_options)
    if not math.isfinite(float(duration_s)) or not .001 <= float(duration_s) <= 2:
        raise ValueError('自动QAM恢复时长须为1～2000 ms')
    sample_limit = min(33554432, max(comp['max_input_samples'], round(duration_s*fs)))
    x, count, _ = _read_span(spectrum, duration_s, None, cancel_event,
                            max_samples=min(math.ceil(2*fs),sample_limit),max_duration_s=2)
    selected = cfg['harmonic']*cfg['base_hz']
    candidates, peaks = _carrier_candidates(x, fs, selected, center_hint_hz)
    # Small search windows keep both-sign/harmonic comparisons bounded.
    screen_n = min(count, max(int(.16*fs),int((len(pwm['symbols'])/pwm['symbol_hz']+len(points)/symbol_hz)*fs)+3))
    screen = x[:screen_n]; time = np.arange(screen_n)/fs
    accepted = []; trials = []
    po = copy.deepcopy(pwm_preamble_options)
    po['search_s'] = min(pwm['search_s'], max(.001,(screen_n-2)/fs))
    for candidate in candidates:
        _cancel(cancel_event)
        row = dict(proposed_carrier_hz=candidate, matched=False)
        try:
            y = screen*np.exp(-2j*np.pi*candidate*time)
            sync = acquire_pwm_header(y, fs, po, cancel_event)
            cfo, _, _, error = _pwm_reference(y, fs, sync, pwm, cancel_event)
            y *= np.exp(-2j*np.pi*cfo*time)
            coefficients, _ = harmonic_coefficients(y,fs,sync['qam_index'],symbol_hz,selected,
                                                    count=len(points),cancel_event=cancel_event)
            fit = fit_training(coefficients,points,symbol_hz,training_bound,tolerance,
                orientation=orientation,fit_dc_offset=dc,estimate_frequency=False,cancel_event=cancel_event)
            row.update(matched=True, carrier_hz=float(candidate+cfo),
                pwm_error_percent=float(error), training_evm_percent=float(fit['evm_percent']),
                header_index=float(sync['index']), conjugated=bool(fit['conjugated']))
            accepted.append(row)
        except ValueError as exc:
            row['reason'] = str(exc)
        trials.append(row)
    if not accepted:
        raise ValueError('自动尝试原载波正负频率及谐波偏移后，未找到同时匹配PWM与QAM训练的候选；请检查RX通道、发端时序和信号')
    accepted.sort(key=lambda v:(v['pwm_error_percent'],v['training_evm_percent']))
    best = accepted[0]
    if any(abs(v['carrier_hz']-best['carrier_hz']) > 2*pwm['max_frequency_hz'] for v in accepted[1:]):
        raise ValueError('多个不同原载波均通过已知前导/训练，存在歧义；请缩小频段或指定通道')
    carrier = best['carrier_hz']
    for start in range(0,len(x),262144):
        _cancel(cancel_event)
        end = min(start+262144,len(x))
        x[start:end] *= np.exp(-2j*np.pi*carrier*np.arange(start,end)/fs)
    ndata = cfg['payload_symbols'] or comp['frame_symbols']
    nominal_frame = len(pwm['symbols'])*fs/pwm['symbol_hz']+ndata*fs/symbol_hz+pwm['gap_s']*fs
    ppm, clock = _known_header_clock(x,fs,pwm,best['header_index'],nominal_frame,
        ndata,symbol_hz,comp['clock_search_ppm'],cancel_event,
        max_headers=max(6,min(20,comp['calibration_frames'])))
    del x
    effective = copy.deepcopy(td)
    effective.setdefault('compensation',{}).update(clock_mode='manual',period_ppm=ppm,
            max_input_samples=sample_limit,phase_mode='pwm_headers')
    from rx706_qam import recover_qam
    result = recover_qam(spectrum,duration_s=duration_s,center_hint_hz=carrier,
        symbol_hz=symbol_hz,order=order,tdcm_options=effective,
        pwm_preamble_options=copy.deepcopy(pwm_preamble_options),
        preamble_options=copy.deepcopy(preamble_options),cancel_event=cancel_event)
    result.diagnostics['auto_acquisition'] = dict(carrier_hz=carrier,period_ppm=ppm,
        spectral_peaks_hz=peaks,candidate_trials=trials,clock=clock,
        selection_source='known_pwm_and_qam_training_only',payload_used=False,
        pwm_observation_average_chips=pwm_preamble_options['observation_average_chips'],
        phase_tracking=result.diagnostics.get('phase_tracking'),
        effective_recovery=dict(duration_s=duration_s,center_hint_hz=carrier,symbol_hz=symbol_hz,
            order=order,tdcm_options=effective,pwm_preamble_options=copy.deepcopy(pwm_preamble_options),
            preamble_options=copy.deepcopy(preamble_options)))
    return result


def main(argv=None):
    """Reproducible read-only capture analysis; outputs only when requested."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from rx706_spectrum import analyze_bin
    from rx706_qam_config import validate_parameter_config
    parser=argparse.ArgumentParser(description='自动原载波+已知前导时钟+QAM训练验收（未知净荷BER未测）')
    parser.add_argument('bin')
    parser.add_argument('--config',required=True)
    parser.add_argument('--channel',type=int,choices=(1,2),default=1)
    parser.add_argument('--duration',type=float,default=2.)
    parser.add_argument('--first-tuple',type=int,default=0)
    parser.add_argument('--carrier-hint',type=float,default=None,help='可选Hz提示，不限制自动候选')
    parser.add_argument('--output-dir',default=None)
    args=parser.parse_args(argv)
    config=validate_parameter_config(json.loads(Path(args.config).read_text()))
    path=Path(args.bin)
    def digest():
        h=hashlib.sha256()
        with path.open('rb') as f:
            while data:=f.read(4*1024*1024):h.update(data)
        return h.hexdigest()
    before=digest()
    spectrum=analyze_bin(path,channel=args.channel,first_tuple=args.first_tuple,nfft=1048576,averages=4)
    r=recover_auto_qam(spectrum,duration_s=args.duration,center_hint_hz=args.carrier_hint,
        **{k:config[k] for k in ('order','symbol_hz','tdcm_options','pwm_preamble_options','preamble_options')})
    if digest()!=before:raise ValueError('输入文件在分析期间改变，不能发布结果')
    a=r.diagnostics['auto_acquisition'];valid=[v for v in r.frames if v['valid']]
    summary=dict(input_path=str(path.resolve()),input_sha256=before,input_unchanged=True,
        channel=args.channel,first_tuple=args.first_tuple,input_samples=r.input_samples,
        sample_rate_hz=spectrum.sample_rate_hz,carrier_hz=a['carrier_hz'],period_ppm=a['period_ppm'],
        frames=len(r.frames),matched_frames=len(valid),payload_candidates=len(r.labels),
        nearest_neighbor_evm_percent=float(r.evm_percent),ber=None,
        training_evm_percent_median=float(np.median([v['pilot_evm_percent'] for v in valid])),
        quality='已知前导和训练已验证；未知净荷仍为候选，BER未测',auto_acquisition=a)
    if args.output_dir:
        out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
        (out/'auto_acquisition_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        (out/'effective_recovery.json').write_text(json.dumps(a['effective_recovery'],ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='auto_acquisition'},ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':main()
