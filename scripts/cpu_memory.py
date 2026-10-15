"""Peak memory of the int8 repack: python3 scripts/cpu_memory.py BATCH short|long (results/cpu-repack-memory.txt)."""
import sys, resource, torch; torch.set_num_threads(1)
sys.path.insert(0, "app"); sys.path.insert(0, "scripts")
from cpu_load import load
from bench import DOC
m = load("models/repack"); m.max_seq_length = 512
bs, n = int(sys.argv[1]), sys.argv[2]
doc = DOC if n == "short" else DOC * 5   # *5 runs past 512 tokens and is truncated at 512
toks = len(m.tokenize([doc])["input_ids"][0])
m.encode([doc] * 32, batch_size=bs)
print(f"batch {bs:2d} doc {toks} tokens: peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MiB")
