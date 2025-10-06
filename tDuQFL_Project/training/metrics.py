
# training/metrics.py
from common.imports import *
from typing import Optional, Dict, Any

# ----------------------------
# Initialization
# ----------------------------
def metrics_init(log_path: str):
    """
    Initialize a CSV-backed metrics store.
    Returns a dict with 'log_path' and in-memory 'rows' (list of rows).
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            # Core federation metrics
            "round","acc_global","acc_clients_mean","acc_clients_std",
            "fairness_gap_p90_p10","time_s","val_loss",
            # Teleportation flags & config
            "use_teleportation","noise","shots","beta",
            # Teleportation QoS (fidelity/latency/instability) summaries
            "fidelity_mean","fidelity_std","fidelity_p10","fidelity_p50","fidelity_p90",
            "latency_mean","latency_std","latency_p90",
            "instability_mean","instability_std",
            # Teleportation update diagnostics
            "delta_norm_dim","delta_l2","delta_max"
        ])
    return {"log_path": log_path, "rows": []}


# ----------------------------
# Per-round logging
# ----------------------------
def _safe_stats(x: Optional[np.ndarray], percentiles=(10, 50, 90)):
    """
    Robust summary: mean, std, and selected percentiles for a 1D array.
    Returns tuple: (mean, std, p10, p50, p90) with Nones if x is empty/None.
    """
    if x is None:
        return (None, None, None, None, None)
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return (None, None, None, None, None)
    mean = float(np.mean(x))
    std  = float(np.std(x, ddof=0))
    pcts = np.percentile(x, percentiles)
    p10, p50, p90 = map(float, pcts)
    return (mean, std, p10, p50, p90)


def metrics_log_round(
    store: Dict[str, Any], *,
    round_idx: int,
    acc_global: float,
    client_accs: Any,
    time_s: float,
    val_loss: Optional[float],
    use_teleportation: bool,
    info: Optional[Dict[str, Any]] = None,
    noise: Optional[str] = None,
    shots: Optional[int] = None
):
    """
    Log a single training round. If teleportation info is available, we record fidelity/latency/instability
    summaries and teleportation update diagnostics.

    Expected (optional) info keys:
      - "fidelity": array-like of per-link/shot fidelities
      - "latency":  array-like of per-link feed-forward latencies
      - "instability": array-like of per-link temporal instabilities
      - "delta_tel": array-like teleportation-conditioned parameter update (for diagnostics)
      - "beta": scalar blending parameter in your aggregation
    """
    # --- client stats & fairness gap ---
    client_accs = np.asarray(client_accs, dtype=float) if client_accs is not None else np.asarray([])
    fair_gap = float(np.percentile(client_accs, 90) - np.percentile(client_accs, 10)) if client_accs.size else None
    acc_mean = float(client_accs.mean()) if client_accs.size else None
    acc_std  = float(client_accs.std(ddof=0)) if client_accs.size else None

    # --- defaults for teleportation-related outputs ---
    beta = None
    fid_mean = fid_std = fid_p10 = fid_p50 = fid_p90 = None
    lat_mean = lat_std = lat_p90 = None
    inst_mean = inst_std = None
    delta_norm_dim = delta_l2 = delta_max = None

    # --- unpack/compute teleportation summaries if available ---
    if use_teleportation and (info is not None):
        # beta (blending)
        if "beta" in info and info["beta"] is not None:
            beta = float(info["beta"])

        # fidelity statistics
        fid = info.get("fidelity", None)
        fid_stats = _safe_stats(None if fid is None else np.asarray(fid))
        fid_mean, fid_std, fid_p10, fid_p50, fid_p90 = fid_stats

        # latency statistics (if provided)
        lat = info.get("latency", None)
        lat_stats = _safe_stats(None if lat is None else np.asarray(lat))
        # We only keep mean, std, and p90 to keep the CSV compact
        lat_mean, lat_std, _, _, lat_p90 = lat_stats

        # instability statistics (if provided)
        inst = info.get("instability", None)
        inst_stats = _safe_stats(None if inst is None else np.asarray(inst))
        inst_mean, inst_std, _, _, _ = inst_stats

        # teleportation update diagnostics
        delta = info.get("delta_tel", None)
        if delta is not None:
            delta = np.asarray(delta).reshape(-1)
            n = max(1, delta.size)
            delta_norm_dim = float(np.linalg.norm(delta) / n)   # per-dimension norm
            delta_l2 = float(np.linalg.norm(delta))
            delta_max = float(np.max(np.abs(delta)))

    # --- construct row ---
    row = [
        int(round_idx),
        float(acc_global), acc_mean, acc_std, fair_gap,
        float(time_s),
        float(val_loss) if val_loss is not None else None,
        bool(use_teleportation),
        str(noise) if noise is not None else None,
        int(shots) if shots is not None else None,
        beta,
        fid_mean, fid_std, fid_p10, fid_p50, fid_p90,
        lat_mean, lat_std, lat_p90,
        inst_mean, inst_std,
        delta_norm_dim, delta_l2, delta_max,
    ]

    # --- append to memory & persist to CSV ---
    store["rows"].append(row)
    with open(store["log_path"], "a", newline="") as f:
        csv.writer(f).writerow(row)

    # --- concise console printout (one line) ---
    _print_round_line(row)


# ----------------------------
# Pretty printing helpers
# ----------------------------
def _fmt(x, nd=3):
    return "NA" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))

def _print_round_line(row):
    (r, acc_g, acc_m, acc_s, fg, tsec, vloss,
     use_tel, noise, shots, beta,
     fmean, fstd, f10, f50, f90,
     lmean, lstd, l90,
     imean, istd,
     dnorm, dl2, dmax) = row

    if use_tel:
        line = (
            f"[Round {r:>3}] acc_g={_fmt(acc_g)} "
            f"(μ={_fmt(acc_m)}, σ={_fmt(acc_s)}, FG={_fmt(fg)}) | "
            f"t={_fmt(tsec)}s, val={_fmt(vloss)} | TEL β={_fmt(beta)}, "
            f"Fid μ={_fmt(fmean)}, σ={_fmt(fstd)}, p50={_fmt(f50)}, p90={_fmt(f90)} | "
            f"Lat μ={_fmt(lmean)}, σ={_fmt(lstd)}, p90={_fmt(l90)} | "
            f"Inst μ={_fmt(imean)}, σ={_fmt(istd)} | "
            f"Δ per-dim={_fmt(dnorm)}, ||Δ||₂={_fmt(dl2)}, max|Δ|={_fmt(dmax)} | "
            f"noise={noise}, shots={shots}"
        )
    else:
        line = (
            f"[Round {r:>3}] acc_g={_fmt(acc_g)} "
            f"(μ={_fmt(acc_m)}, σ={_fmt(acc_s)}, FG={_fmt(fg)}) | "
            f"t={_fmt(tsec)}s, val={_fmt(vloss)} | TEL=FALSE"
        )
    print(line)


# ----------------------------
# Finalization & summaries
# ----------------------------
def metrics_finalize(store):
    """
    Return the in-memory rows as a numpy object array.
    """
    return np.array(store["rows"], dtype=object)


def compute_auc(acc_list):
    """
    Simple area-under-curve over rounds (trapezoidal rule),
    with x = round indices [0,1,2,...].
    """
    y = np.asarray(acc_list, dtype=float)
    x = np.arange(len(y))
    return float(np.trapz(y, x))


def metrics_summarize(store: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce an end-of-run summary (useful for console or saving to JSON).
    Returns a dict; nothing is written to disk.
    """
    rows = store.get("rows", [])
    if not rows:
        return {}

    arr = np.array(rows, dtype=object)

    # Column indices (aligned with header order in metrics_init)
    COL = {
        "round": 0, "acc_global": 1, "acc_clients_mean": 2, "acc_clients_std": 3,
        "fair_gap": 4, "time_s": 5, "val_loss": 6,
        "use_tel": 7, "noise": 8, "shots": 9, "beta": 10,
        "fid_mean": 11, "fid_std": 12, "fid_p10": 13, "fid_p50": 14, "fid_p90": 15,
        "lat_mean": 16, "lat_std": 17, "lat_p90": 18,
        "inst_mean": 19, "inst_std": 20,
        "delta_norm_dim": 21, "delta_l2": 22, "delta_max": 23,
    }

    def _col_numeric(ix):
        vals = [v for v in arr[:, ix].tolist() if isinstance(v, (int, float))]
        return np.asarray(vals, dtype=float) if len(vals) else None

    acc_g = _col_numeric(COL["acc_global"]) or np.array([], dtype=float)
    fair  = _col_numeric(COL["fair_gap"])
    tel_mask = np.array([bool(v) for v in arr[:, COL["use_tel"]].tolist()])

    # Global aggregates
    out = {
        "rounds": int(arr.shape[0]),
        "acc_global_last": float(acc_g[-1]) if acc_g.size else None,
        "acc_global_best": float(np.max(acc_g)) if acc_g.size else None,
        "acc_global_auc": compute_auc(acc_g) if acc_g.size else None,
        "fairness_gap_mean": float(np.mean(fair)) if fair is not None and fair.size else None,
        "teleport_rounds": int(np.sum(tel_mask)),
        "nonteleport_rounds": int(arr.shape[0] - np.sum(tel_mask)),
    }

    # Teleportation-only aggregates
    if np.any(tel_mask):
        def tele_stat(name):
            vec = _col_numeric(COL[name])
            if vec is None or not vec.size:
                return None
            # keep only rows where use_tel==True
            tvals = [arr[i, COL[name]] for i in range(arr.shape[0]) if tel_mask[i] and isinstance(arr[i, COL[name]], (int, float))]
            if not tvals:
                return None
            tv = np.asarray(tvals, dtype=float)
            return {"mean": float(np.mean(tv)), "std": float(np.std(tv, ddof=0)),
                    "p50": float(np.percentile(tv, 50)), "p90": float(np.percentile(tv, 90))}

        out["teleport_fidelity"]   = tele_stat("fid_mean")
        out["teleport_latency"]    = tele_stat("lat_mean")
        out["teleport_instability"]= tele_stat("inst_mean")
        out["teleport_beta"]       = tele_stat("beta")
        out["teleport_delta_l2"]   = tele_stat("delta_l2")

    # Print a compact summary for convenience
    print("\n=== Training Summary ===")
    print(f"Rounds: {out['rounds']} | TEL rounds: {out['teleport_rounds']} | Non-TEL: {out['nonteleport_rounds']}")
    print(f"Acc_global: last={_fmt(out['acc_global_last'])}, best={_fmt(out['acc_global_best'])}, AUC={_fmt(out['acc_global_auc'])}")
    print(f"Fairness gap (p90-p10): mean={_fmt(out['fairness_gap_mean'])}")
    if out.get("teleport_fidelity"):
        tf = out["teleport_fidelity"]
        print(f"Teleportation fidelity μ={_fmt(tf['mean'])} (σ={_fmt(tf['std'])}, p50={_fmt(tf['p50'])}, p90={_fmt(tf['p90'])})")
    if out.get("teleport_latency"):
        tl = out["teleport_latency"]
        print(f"Teleportation latency μ={_fmt(tl['mean'])} (σ={_fmt(tl['std'])}, p50={_fmt(tl['p50'])}, p90={_fmt(tl['p90'])})")
    if out.get("teleport_instability"):
        ti = out["teleport_instability"]
        print(f"Teleportation instability μ={_fmt(ti['mean'])} (σ={_fmt(ti['std'])}, p50={_fmt(ti['p50'])}, p90={_fmt(ti['p90'])})")
    if out.get("teleport_beta"):
        tb = out["teleport_beta"]
        print(f"β summary μ={_fmt(tb['mean'])} (σ={_fmt(tb['std'])}, p50={_fmt(tb['p50'])}, p90={_fmt(tb['p90'])})")
    if out.get("teleport_delta_l2"):
        td = out["teleport_delta_l2"]
        print(f"‖Δ_tel‖₂ μ={_fmt(td['mean'])} (σ={_fmt(td['std'])}, p50={_fmt(td['p50'])}, p90={_fmt(td['p90'])})")

    return out

'''
def metrics_init(log_path):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "round","acc_global","acc_clients_mean","acc_clients_std",
            "fairness_gap_p90_p10","time_s",
            "val_loss","use_teleportation","noise","shots",
            "fidelity_mean","fidelity_std","delta_norm_dim","beta"
        ])
    return {"log_path": log_path, "rows": []}

def metrics_log_round(store, *, round_idx, acc_global, client_accs, time_s,
                      val_loss, use_teleportation, info=None, noise=None, shots=None):
    client_accs = np.asarray(client_accs, dtype=float)
    fair_gap = float(np.percentile(client_accs, 90) - np.percentile(client_accs, 10)) if client_accs.size else None
    if use_teleportation and (info is not None):
        fid = np.asarray(info["fidelity"]).reshape(-1)
        delta = np.asarray(info["delta_tel"]).reshape(-1)
        row = [
            round_idx, float(acc_global),
            float(client_accs.mean()) if client_accs.size else None,
            float(client_accs.std(ddof=0)) if client_accs.size else None,
            fair_gap, float(time_s),
            float(val_loss) if val_loss is not None else None,
            True, str(noise) if noise is not None else None,
            int(shots) if shots is not None else None,
            float(fid.mean()), float(fid.std(ddof=0)),
            float(np.linalg.norm(delta)/max(1, delta.size)),
            float(info["beta"]),
        ]
    else:
        row = [round_idx, float(acc_global),
               float(client_accs.mean()) if client_accs.size else None,
               float(client_accs.std(ddof=0)) if client_accs.size else None,
               fair_gap, float(time_s),
               float(val_loss) if val_loss is not None else None,
               False, None, None, None, None, None, None]
    store["rows"].append(row)
    with open(store["log_path"], "a", newline="") as f:
        csv.writer(f).writerow(row)

def metrics_finalize(store):
    return np.array(store["rows"], dtype=object)

def compute_auc(acc_list):
    y = np.asarray(acc_list, dtype=float)
    x = np.arange(len(y))
    return float(np.trapz(y, x))
'''