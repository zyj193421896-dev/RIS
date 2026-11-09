"""Read-only mapping/binary/model audit of exported 64QAM codebooks; no GUI/IQ claim."""
from pathlib import Path
import argparse,csv,hashlib,io,json,math,sys,zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam import decide
from rx706_qam_preamble import points_from_labels
from rx706_qam_training import fit_training,normalize_training
from rx706_tdcm import harmonic_coefficients
from rx706_qam_profiles import four_corner_labels


def audit(path):
    with zipfile.ZipFile(path) as z:
        meta=json.loads(z.read('metadata.json'));cfg=meta['config']
        assert cfg['qam_order']==64 and meta['shape'][1]==64
        for line in z.read('SHA256SUMS.txt').decode().splitlines():
            digest,name=line.split(maxsplit=1);assert hashlib.sha256(z.read(name.strip().lstrip('*'))).hexdigest()==digest
        symbols=list(csv.DictReader(io.StringIO(z.read('symbols.csv').decode('utf-8-sig'))))
        labels=np.array([int(r['symbol']) for r in symbols]);assert np.array_equal(labels,np.arange(64))
        xy=np.array([complex(float(r['normalized_real']),float(r['normalized_imag'])) for r in symbols])
        for row in symbols:
            i=int(row['symbol']);assert row['bits_IIIQQQ']==f'{i:06b}' and int(row['hex_byte'],16)==i
        expected=points_from_labels(labels,64);ratio=7*math.sqrt(2)/math.sqrt(42)
        np.testing.assert_allclose(xy*ratio,expected)
        nd,_,slots,units=meta['shape'];frames=np.frombuffer(z.read('codebook_frames.bin'),dtype=np.uint8)
        frame_bytes=meta['binary_layout']['frame_bytes'];assert frames.size==nd*64*slots*frame_bytes
        packed=frames.reshape(nd,64,slots,frame_bytes);states=np.unpackbits(packed,axis=-1,bitorder='little')
        assert np.all(states[:,:,:,units:]==0);states=states[:,:,:,:units]
        rows=list(csv.DictReader(io.StringIO(z.read('pulse_parameters.csv').decode())))
        width=np.zeros((nd,64,units),int);start=np.zeros_like(width)
        for r in rows:
            index=(int(r['direction']),int(r['symbol']),int(r['unit']));width[index]=int(r['width_slots']);start[index]=int(r['start_slot'])
        rule=((np.arange(slots)[None,None,:,None]-start[:,:,None,:])%slots)<width[:,:,None,:]
        assert np.array_equal(states,rule)
        # Independently reconstruct the documented near/far propagation model.
        unit=np.arange(units);x=(unit%cfg['cols']-(cfg['cols']-1)/2)*cfg['col_pitch_mm']*.001
        y=(unit//cfg['cols']-(cfg['rows']-1)/2)*cfg['row_pitch_mm']*.001
        pos=np.column_stack([x,y,np.zeros(units)]);tx=np.array([cfg['tx_x_m'],cfg['tx_y_m'],cfg['tx_z_m']])
        rt=np.linalg.norm(pos-tx,axis=1);rtref=np.linalg.norm(tx);h=cfg['harmonic'];base=cfg['pwm_mhz']*1e6
        kin=2*np.pi*cfg['carrier_ghz']*1e9/299792458.;kout=2*np.pi*(cfg['carrier_ghz']*1e9+h*base)/299792458.
        gamma0=cfg['gamma0_amplitude']*np.exp(1j*np.deg2rad(cfg['gamma0_phase_deg']))
        gamma1=cfg['gamma1_amplitude']*np.exp(1j*np.deg2rad(cfg['gamma1_phase_deg']))
        models=[]
        for d,angle in enumerate(cfg['scan_angles_deg']):
            theta=np.deg2rad(angle);az=np.deg2rad(cfg['scan_azimuth_deg']);direction=np.array([np.sin(theta)*np.cos(az),np.sin(theta)*np.sin(az),np.cos(theta)])
            if cfg['rx_mode']=='near':
                rr=np.linalg.norm(pos-cfg['rx_range_m']*direction,axis=1);phase=kin*(rt-rtref)+kout*(rr-cfg['rx_range_m']);weights=1/(rt*rr)
            else:phase=kin*(rt-rtref)-kout*(pos@direction);weights=1/rt
            propagation=weights*np.exp(-1j*phase)
            wave=(gamma0+(gamma1-gamma0)*states[d])@propagation
            periods=cfg['repeats'];wave=np.tile(wave[:,None,:],(1,periods,1)).reshape(-1)
            coeff,_=harmonic_coefficients(wave,base*slots,0,base/periods,h*base,count=64)
            observed=coeff/(weights.sum()*abs(gamma1-gamma0)/np.pi)
            target=xy*cfg['amplitude_scale'];gain=np.vdot(target,observed)/np.vdot(target,target)
            evm=float(100*np.sqrt(np.mean(abs(observed/gain-target)**2)/np.mean(abs(target)**2)))
            assert abs(evm-meta['directions'][d]['evm_equalized_percent'])<1e-8
            corners=four_corner_labels(64);fit=fit_training(observed[corners],expected[corners],base/periods,1000,10,orientation='auto',fit_dc_offset=True)
            corrected=normalize_training(observed,fit);decisions=decide(corrected,64)[1]
            models.append(dict(direction=d,model_reference_evm_percent=evm,
                hypothetical_corner_calibration_symbol_errors=int(np.count_nonzero(decisions!=labels)),
                hypothetical_corner_evm_percent=fit['evm_percent']))
        return dict(input=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            symbols=64,bits_per_symbol=6,corners_hex=[f'{v:02X}' for v in four_corner_labels(64)],slots=slots,
            pwm_hz=base,repeats=cfg['repeats'],symbol_hz=meta['symbol_rate_baud'],gross_bit_rate_bps=meta['gross_bit_rate_bps'],
            binary_and_checksums_valid=True,models=models,contains_training_frame=False,
            note='码本是查找表；没有PWM/四角发送帧。模型投影和假设四角标定均非真实IQ/BER测量。')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('codebooks',nargs='+',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();result=[audit(path) for path in a.codebooks];text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:f.write(text)
    print(text)

if __name__=='__main__':main()
