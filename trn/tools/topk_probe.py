import os, sys, subprocess, time, torch
os.environ.setdefault("PJRT_DEVICE", "CPU")
import torch_xla
dev = torch_xla.device()
OUT = "/opt/hyt/isgv"
FLAGS = ["--target=trn2", "--model-type=transformer", "--optlevel=1", "--enable-saturate-infinity", "--lnc=2", "--verbose=35"]

def compile_fn(name, fn, *shapes, dtype=torch.float32):
    torch.manual_seed(0)
    ins = [torch.randn(*s, device=dev).to(dtype) for s in shapes]
    torch_xla.sync()
    out = fn(*ins)
    outs = list(out) if isinstance(out, (tuple, list)) else [out]
    pb = torch_xla._XLAC._get_xla_tensors_hlo_proto(outs)
    pbf = f"{OUT}/probe_{name}.hlo.pb"
    open(pbf, "wb").write(pb)
    t0 = time.time()
    r = subprocess.run(["neuronx-cc", "compile", "--framework=XLA", pbf, "--output", pbf.replace(".hlo.pb", ".neff")] + FLAGS, capture_output=True, text=True, timeout=3600)
    o = r.stdout + r.stderr
    tag = "OK" if r.returncode == 0 else ("ISGV902" if "ISGV902" in o else f"FAIL rc={r.returncode}: " + o[-400:].replace("\n", " | "))
    print(f"[{name}] {tag} in {time.time()-t0:.0f}s", flush=True)

NT, M = 4096, 32
which = sys.argv[1]
if which == "widths":
    for W in (512, 2048, 4096, 8192, 16384, 32768):
        compile_fn(f"topk_w{W}_k32", lambda x: torch.topk(x, M, dim=1, sorted=True)[1], (NT, W))
elif which == "ks":
    for k in (1, 8, 16, 64, 128):
        compile_fn(f"topk_w32768_k{k}", lambda x, k=k: torch.topk(x, k, dim=1, sorted=True)[1], (NT, 32768))
    compile_fn("topk_w32768_k32_unsorted", lambda x: torch.topk(x, 32, dim=1, sorted=False)[1], (NT, 32768))
    compile_fn("topk_w32768_k32_bf16", lambda x: torch.topk(x, 32, dim=1, sorted=True)[1], (NT, 32768), dtype=torch.bfloat16)
    compile_fn("topk_w32768_k32_nt512", lambda x: torch.topk(x, 32, dim=1, sorted=True)[1], (512, 32768))
elif which == "alts":
    def two_stage(x):  # exact top-M: per-slot top-M (width 512) then top-M over the S*M survivors
        S, P = 64, 512
        xs = x.reshape(NT, P, S).permute(0, 2, 1)             # [nt, S, P], f = p*S + s
        v1, i1 = torch.topk(xs, M, dim=2, sorted=False)         # [nt, S, M] (p indices)
        v2, i2 = torch.topk(v1.reshape(NT, S * M), M, dim=1, sorted=True)
        s = i2 // M
        p = torch.gather(i1.reshape(NT, S * M), 1, i2)
        return (p * S + s).to(torch.int32)
    compile_fn("two_stage", two_stage, (NT, 32768))
    def two_stage_from_dot(x):  # from the [nt,S,P] dot output directly, no big transpose
        S, P = 64, 512
        xs = x                                                  # [nt, S, P]
        v1, i1 = torch.topk(xs, M, dim=2, sorted=False)
        v2, i2 = torch.topk(v1.reshape(NT, S * M), M, dim=1, sorted=True)
        s = i2 // M
        p = torch.gather(i1.reshape(NT, S * M), 1, i2)
        return (p * S + s).to(torch.int32)
    compile_fn("two_stage_nsp", two_stage_from_dot, (NT, 64, 512))
    compile_fn("sort_top32", lambda x: torch.sort(x, dim=1, descending=True)[1][:, :M].to(torch.int32), (NT, 32768))
    compile_fn("argmax_only", lambda x: torch.argmax(x, dim=1), (NT, 32768))
if which == "alts2":
    S, P = 64, 512
    def two_stage_2d(x):  # x: [nt, S, P] dot output; exact global top-M; only 2D sorted topk
        v1, i1 = torch.topk(x.reshape(NT * S, P), M, dim=1, sorted=True)      # [nt*S, M]
        v2, i2 = torch.topk(v1.reshape(NT, S * M), M, dim=1, sorted=True)     # [nt, M]
        s = i2 // M
        p = torch.gather(i1.reshape(NT, S * M), 1, i2)
        return (p * S + s).to(torch.int32)
    compile_fn("two_stage_2d", two_stage_2d, (NT, S, P))
    def full_proposer(dh, cb):  # dot + two-stage, the candidate production code
        D = 10
        lv = dh.reshape(NT, S, D).to(torch.bfloat16).to(torch.float32)
        cq = cb.to(torch.bfloat16).to(torch.float32)
        nh = cq / (cq.square().sum(1, keepdim=True).sqrt() + 2**-14)
        nh_g = nh.reshape(P, S, D).permute(1, 0, 2)
        sc = torch.einsum("nsd,spd->nsp", lv, nh_g)                             # [nt, S, P]
        return two_stage_2d(sc)
    compile_fn("full_proposer_2stage", full_proposer, (NT, S * 10), (32768, 10))
    def half_split(x):  # alternative: two topk over 16384 halves then merge
        a_v, a_i = torch.topk(x[:, :16384], M, dim=1, sorted=True)
        b_v, b_i = torch.topk(x[:, 16384:], M, dim=1, sorted=True)
        v, i = torch.topk(torch.cat([a_v, b_v], 1), M, dim=1, sorted=True)
        idx = torch.cat([a_i, b_i + 16384], 1)
        return torch.gather(idx, 1, i).to(torch.int32)
    compile_fn("half_split", half_split, (NT, 32768))
