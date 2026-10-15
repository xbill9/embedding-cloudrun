# The Short Answer

A basic 1 vCPU CPU service is good enough, and it is the cheapest option for most of what an embedding service does. Add the L4 only for big batch jobs.

- **Minimum spec:** 1 vCPU, 2 GiB, int8 weights. Queries take a median 133 ms, a million documents cost USD 27.67, and an idle service costs nothing. 1 GiB is too small. Documents near 512 tokens need a batch size of 8 or less, or 4 GiB.
- **The L4 bills its idle time.** Each isolated burst was billed about eight minutes of L4 time, USD 0.14, even when the work took seconds. The L4 wins once a single burst passes about 5,300 documents: a million documents cost USD 1.83 on it against USD 27.67 on the CPU.
- **An L4 kept warm costs USD 764 a month.** The CPU service embeds 27.6 million documents for the same money.
- **The int4 repack is slower on CPU.** The file is 28% smaller, but queries take 1.52x as long and it needs 3 GiB, because this server runs it in float32. Retrieval quality is about the same: SciFact nDCG@10 of 0.834 against 0.836 for int8.

In practice: serve queries and small indexing batches from the 1 vCPU int8 service. For a job of tens of thousands of documents, deploy the L4, run the job, and let it scale to zero.

