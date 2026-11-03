"""Generate the fault dataset in parallel.  Usage: python src/generate.py --per-topo 6000"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sim import Simulator, TOPOLOGIES, sample_class

def work(args):
    topo, n, seed = args
    sim = Simulator(seed)
    rows, V0, V1, I0, I1 = [], [], [], [], []
    while len(rows) < n:
        s = sim.sample(topo, sample_class(sim.rng))
        if s is None:
            continue
        rows.append(dict(topo=s.topo, ftype=s.ftype, zone=s.zone, bus=s.bus, phases=s.phases,
                         rf=s.rf, load_mult=s.load_mult, pv_pen=s.pv_pen, cap_on=s.cap_on))
        V0.append(s.v_pre); V1.append(s.v_post); I0.append(s.i_pre); I1.append(s.i_post)
    return rows, np.array(V0), np.array(V1), np.array(I0), np.array(I1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topo", type=int, default=6000)
    ap.add_argument("--chunks", type=int, default=6, help="jobs per topology")
    ap.add_argument("--out", default="data/raw_phasors.npz")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    jobs = []
    for ti, t in enumerate(TOPOLOGIES):
        per = a.per_topo // a.chunks
        for c in range(a.chunks):
            jobs.append((t, per, 1000 * ti + c))
    t0 = time.time()
    res = []
    with Pool(a.workers) as p:
        for k, r in enumerate(p.imap(work, jobs), 1):
            res.append(r)
            print(f"[{k}/{len(jobs)}] chunks done, {time.time()-t0:.0f}s", flush=True)
    meta = pd.DataFrame([row for r in res for row in r[0]])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, v_pre=np.concatenate([r[1] for r in res]), v_post=np.concatenate([r[2] for r in res]),
                        i_pre=np.concatenate([r[3] for r in res]), i_post=np.concatenate([r[4] for r in res]))
    meta.to_csv(a.out.replace(".npz", "_meta.csv"), index=False)
    print(meta.groupby(["topo", "ftype"]).size().unstack())
    print(f"saved {len(meta)} samples in {time.time()-t0:.0f}s")
