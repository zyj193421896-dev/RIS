"""Explicit three-user PWM CRC protocol; no reference-assisted correction.

score_slots consumes scheduled physical slots, not only detected headers. The
caller must supply missing slots when its continuous capture time axis permits
counting them. No frame number exists in this protocol: unknown capture gaps
cannot be reconstructed from the repeating payload.
"""
import math
from numbers import Integral

HEADERS = ((0, 7, 7, 0), (0, 0, 7, 7), (7, 0, 0, 7))
PAYLOADS = (tuple(range(8)), tuple(reversed(range(8))), (0, 2, 4, 6, 1, 3, 5, 7))
CALIBRATION = (0, 7, 0, 7)
CRC_PARAMETERS = dict(width=12, polynomial=0x80F, init=0, xorout=0,
                      refin=False, refout=False, bits_per_symbol=3,
                      covered_symbols=16, crc_symbols=4)


def crc12(symbols):
    """Direct MSB-first update, no additional zero bits, no final reflection."""
    register = 0
    for symbol in symbols:
        if isinstance(symbol, bool) or not isinstance(symbol, Integral) or not 0 <= symbol < 8:
            raise ValueError('CRC input must contain integer symbols 0..7')
        symbol = int(symbol)
        for shift in (2, 1, 0):
            feedback = ((register >> 11) & 1) ^ ((symbol >> shift) & 1)
            register = (register << 1) & 0xFFF
            if feedback:
                register ^= 0x80F
    return register


def append_crc(symbols):
    symbols = tuple(symbols)
    if len(symbols) != 16:
        raise ValueError('CRC frame must contain 16 covered symbols')
    value = crc12(symbols)
    return symbols + tuple((value >> shift) & 7 for shift in (9, 6, 3, 0))


def test_frames():
    return tuple(append_crc(h + p + CALIBRATION) for h, p in zip(HEADERS, PAYLOADS))


def evaluate_slot(record):
    """None denotes erased received symbol. Never insert expected symbols."""
    uid = record['user_id']
    if isinstance(uid, bool) or not isinstance(uid, Integral) or uid not in (1, 2, 3):
        raise ValueError('user_id must be 1, 2, or 3')
    uid = int(uid)
    complete = bool(record.get('complete', False))
    symbols = record.get('received_symbols')
    if symbols is None:
        symbols = [None] * 20
    if len(symbols) != 20:
        raise ValueError('received_symbols must preserve all 20 slots using None for erasures')
    valid = [isinstance(s, Integral) and not isinstance(s, bool) and 0 <= s < 8 for s in symbols]
    if any(s is not None and not ok for s, ok in zip(symbols, valid)):
        raise ValueError('received symbols must be integer 0..7 or None')
    symbols = [int(s) if ok else None for s, ok in zip(symbols, valid)]
    identity_ok = bool(record.get('identity_ok', False)) and tuple(symbols[:4]) == HEADERS[uid-1]
    calibration_ok = bool(record.get('calibration_ok', False)) and tuple(symbols[12:16]) == CALIBRATION
    crc_ok = crc12(symbols) == 0 if all(valid) else None
    erased = not all(valid)
    failed = complete and (not identity_ok or not calibration_ok or erased or crc_ok is not True)
    return dict(user_id=uid, complete=complete,
                detected=bool(record.get('detected', any(valid))),
                expected=bool(record.get('expected', True)),
                identity_ok=identity_ok, calibration_ok=calibration_ok,
                erased=erased, crc_ok=crc_ok, failed=bool(failed),
                received_symbols=list(symbols), payload_symbols=list(symbols[4:12]))


def _summary(rows, continuous):
    complete = [r for r in rows if r['complete'] and r['expected']]
    n = len(complete)
    failures = sum(r['failed'] for r in complete)
    return dict(expected_complete_slots=n, detected_complete_slots=sum(r['detected'] for r in complete),
                partial_slots=sum(not r['complete'] for r in rows),
                failed_slots=failures, passed_slots=n-failures, crc_failures=sum(r['crc_ok'] is False for r in complete),
                crc_untestable=sum(r['crc_ok'] is None for r in complete),
                identity_failures=sum(not r['identity_ok'] for r in complete),
                calibration_failures=sum(not r['calibration_ok'] for r in complete),
                erased_slots=sum(r['erased'] for r in complete),
                observed_scheduled_bler=failures/n if n else None,
                continuous_window_bler=failures/n if n and continuous else None,
                transmission_bler=None,
                metric_caveat='CRC detects errors but does not establish payload ground truth; valid CRC may miss errors',
                confidence_caveat='Zero-failure bound assumes independent identically distributed sampled slots; applies only to supplied capture-window slots, not the entire transmission',
                zero_failure_upper95=(1-math.exp(math.log(.05)/n)) if n and not failures else None)


def score_slots(records, timeline_contiguous=False):
    """Observed window metric only; capture completeness is never inferred.

    timeline_contiguous certifies all scheduled physical slots inside the scored
    window were supplied, including silence. Even then end-to-end transmission
    BLER remains unknown outside that window. Independence is required for the
    optional zero-failure confidence bound.
    """
    rows = [evaluate_slot(r) for r in records]
    return dict(protocol='three-user-pwm-crc12-v1', timeline_contiguous=bool(timeline_contiguous),
                scope='continuous scheduled capture window' if timeline_contiguous else 'supplied scheduled slots; unknown gaps excluded',
                missing_whole_cycles_identifiable=bool(timeline_contiguous),
                overall=_summary(rows, timeline_contiguous),
                users={str(uid): _summary([r for r in rows if r['user_id'] == uid], timeline_contiguous)
                       for uid in (1, 2, 3)}, slots=rows)
