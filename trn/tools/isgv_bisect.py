"""Compile-only bisection of ISGV902: trace proposer variants under PJRT_DEVICE=CPU,
dump HloModuleProto, run neuronx-cc with the production flags. No NeuronCore needed."""
import os, sys, subprocess, time, torch
os.environ.setdefault("PJRT_DEVICE", "CPU")
import torch_xla
import torch_xla.core.xla_model as xm

NT, S, D, NF, M = int(os.environ.get("NT", 4096)), 64, int(os.environ.get("DSLOT", 10)), 32768, 32
P = NF // S
dev = torch_xla.device()
OUT = "/opt/hyt/isgv"
FLAGS = ["--target=trn2", "--model-type=transformer", "--optlevel=1", "--enable-saturate-infinity", "--lnc=2", "--verbose=35"]

def nhat_of(cb):
    cq = cb.to(torch.bfloat16).to(torch.float32)
    return cq / (cq.square().sum(1, keepdim=True).sqrt() + 2**-14)

def v_current(dh, cb):
    lv = dh.reshape(NT, S, D).to(torch.bfloat16).to(torch.float32)
    nh_g = nhat_of(cb).reshape(P, S, D).permute(1, 0, 2)
    sc = torch.einsum("nsd,spd->nsp", lv, nh_g)
    scores = sc.permute(0, 2, 1).reshape(NT, NF)
    return torch.topk(scores, M, dim=1, largest=True, sorted=True)[1].to(torch.int32)

def v_no_topk(dh, cb):
    lv = dh.reshape(NT, S, D).to(torch.bfloat16).to(torch.float32)
    nh_g = nhat_of(cb).reshape(P, S, D).permute(1, 0, 2)
    sc = torch.einsum("nsd,spd->nsp", lv, nh_g)
    return sc.permute(0, 2, 1).reshape(NT, NF)

def v_dot_only(dh, cb):
    lv = dh.reshape(NT, S, D).to(torch.bfloat16).to(torch.float32)
    nh_g = nhat_of(cb).reshape(P, S, D).permute(1, 0, 2)
    return torch.einsum("nsd,spd->nsp", lv, nh_g)

def v_topk_only(dh, cb):
    scores = dh.reshape(NT, NF)  # requires dh of size NT*NF; handled below
    return torch.topk(scores, M, dim=1, largest=True, sorted=True)[1].to(torch.int32)

def v_blockdiag(dh, cb):
    # scores[t, f] = <leaf[t, f%S], nhat[f]> as ONE dense matmul [NT, S*D] x [S*D, NF]
    lv = dh.reshape(NT, S * D).to(torch.bfloat16).to(torch.float32)
    nh = nhat_of(cb)                                            # [NF, D]
    f = torch.arange(NF, device=dh.device)
    mask = (f.remainder(S).unsqueeze(0) == torch.arange(S, device=dh.device).unsqueeze(1)).to(torch.float32)  # [S, NF]
    W = (nh.t().unsqueeze(0) * mask.unsqueeze(1)).reshape(S * D, NF)  # [S*D, NF]
    scores = lv @ W
    return torch.topk(scores, M, dim=1, largest=True, sorted=True)[1].to(torch.int32)

def v_blockdiag_bf16(dh, cb):
    lv = dh.reshape(NT, S * D).to(torch.bfloat16)
    nh = nhat_of(cb)
    f = torch.arange(NF, device=dh.device)
    mask = (f.remainder(S).unsqueeze(0) == torch.arange(S, device=dh.device).unsqueeze(1)).to(torch.float32)
    W = (nh.t().unsqueeze(0) * mask.unsqueeze(1)).reshape(S * D, NF).to(torch.bfloat16)
    scores = (lv @ W).to(torch.float32)
    return torch.topk(scores, M, dim=1, largest=True, sorted=True)[1].to(torch.int32)

VARIANTS = {"current": v_current, "no_topk": v_no_topk, "dot_only": v_dot_only,
            "topk_only": v_topk_only, "blockdiag": v_blockdiag, "blockdiag_bf16": v_blockdiag_bf16}

def run(name):
    fn = VARIANTS[name]
    torch.manual_seed(0)
    if name == "topk_only":
        dh = torch.randn(NT, NF, device=dev)
    else:
        dh = torch.randn(NT, S * D, device=dev)
    cb = torch.randn(NF, D, device=dev)
    xm.mark_step()
    out = fn(dh, cb)
    pb = torch_xla._XLAC._get_xla_tensors_hlo_proto([out])
    pbf = f"{OUT}/{name}_nt{NT}_d{D}.hlo.pb"
    open(pbf, "wb").write(pb if isinstance(pb, (bytes, bytearray)) else pb.encode())
    t0 = time.time()
    r = subprocess.run(["neuronx-cc", "compile", "--framework=XLA", pbf, "--output", pbf.replace(".hlo.pb", ".neff")] + FLAGS,
                       capture_output=True, text=True, timeout=3600)
    dt = time.time() - t0
    tag = "OK" if r.returncode == 0 else ("ISGV902" if "ISGV902" in (r.stdout + r.stderr) else f"FAIL rc={r.returncode}")
    print(f"[{name} nt={NT} d={D}] {tag} in {dt:.0f}s", flush=True)
    if r.returncode != 0 and tag != "ISGV902":
        print((r.stdout + r.stderr)[-1500:])
    return tag


def v_two_stage_key(dh, cb):
    lv = dh.reshape(NT, S, D).to(torch.bfloat16).to(torch.float32)
    nh_g = nhat_of(cb).reshape(P, S, D).permute(1, 0, 2)
    sc = torch.einsum("nsd,spd->nsp", lv, nh_g)
    m1 = 32
    k1 = sc - torch.arange(P, device=dh.device, dtype=torch.float32) * 2.0 ** -40
    v1, p1 = torch.topk(k1.reshape(NT * S, P), m1, dim=1, largest=True, sorted=True)
    v1 = v1.reshape(NT, S, m1).transpose(1, 2); p1 = p1.reshape(NT, S, m1).transpose(1, 2)
    k2 = v1 - torch.arange(S, device=dh.device, dtype=torch.float32) * 2.0 ** -50
    _, j = torch.topk(k2.reshape(NT, m1 * S), M, dim=1, largest=True, sorted=True)
    s = j % S
    p = torch.gather(p1.reshape(NT, m1 * S), 1, j)
    return (p * S + s).to(torch.int32)
VARIANTS["two_stage_key"] = v_two_stage_key
if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
