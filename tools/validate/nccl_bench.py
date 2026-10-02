# All-reduce latency/bandwidth between the two serving nodes, on the same torch + NCCL
# that vLLM uses. Launched by nccl_bench.sbatch (one rank per node).
#   decode-sized messages (hidden size x a few tokens, bf16) are what TP2 pays per layer;
#   large messages show link bandwidth (prefill).
import os, time, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(0)
sizes = [8 << 10, 32 << 10, 128 << 10, 512 << 10, 2 << 20, 8 << 20, 32 << 20, 128 << 20]
for graph in (False, True):
    for nbytes in sizes:
        x = torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda")
        iters = 200 if nbytes <= (2 << 20) else 30
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        if graph:  # vLLM captures decode in CUDA graphs; launch overhead disappears there
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                dist.all_reduce(x)
                torch.cuda.synchronize()
                with torch.cuda.graph(g):
                    for _ in range(iters):
                        dist.all_reduce(x)
            torch.cuda.synchronize()
            t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        else:
            t = time.perf_counter()
            for _ in range(iters):
                dist.all_reduce(x)
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / iters
        if rank == 0:
            busbw = nbytes / dt / 1e9  # 2 ranks: algbw == busbw
            print(f"NCCL graph={int(graph)} bytes={nbytes:>10} lat_us={dt*1e6:9.1f} busbw_GBps={busbw:7.2f}", flush=True)
dist.destroy_process_group()
