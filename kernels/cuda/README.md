# kernels/cuda

Device kernels gated by the differential suite (spec §04). **No kernel enters
the harness until it matches `libapply_ref.so` bit for bit on the exact GPU
architecture of the citable run** — passing on Ada does not authorize Hopper
or Blackwell (per-silicon gate, plan M3).

- `refkernels.cu` — normalize (Ĉ), pack+allocate (top-k global, winner per
  slot, abort rules on bf16 bits), apply (±0 no-op by rule). `-fmad=false`
  mandatory.
- `diff_harness.cu` — THE GATE: loads `libapply_ref.so`, generates adversarial
  families (leaves salted with `-0.0`, denormals, NaN/inf, zero delta_hat for
  step-0 dynamics, dense random) and compares verdict streams and residuals
  bit for bit. Exit 0 = `BIT_IDENTICAL`, 1 = `BACKEND_REJECTED`.

Build + run (on the run droplet):

```
nvcc -O2 -fmad=false -arch=native diff_harness.cu refkernels.cu -o diff_harness -ldl
./diff_harness /opt/hytorch/target/release/libapply_ref.so 200000
```
