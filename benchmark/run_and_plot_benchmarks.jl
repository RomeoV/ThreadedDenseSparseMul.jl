"""
    Summary of results:

For matrices (NxK) and (KxM) we fix N=1_000 and K=2_000, and vary N.
For all N we see a speed up, up to approximately 3x for N=30_000.
"""

using SparseArrays, SparseMatricesCSR, ThreadedSparseCSR
using Plots, StatsPlots, BenchmarkPlots, BenchmarkTools
using DenseSparseMul  # note that this overwrites the regular mul!
import SparseArrays: mul!, findnz
# import ThreadPinning: pinthreads
# pinthreads(:cores)

suite = BenchmarkGroup()
N, K, M = 1_000, 2_000, 30_000;

suite[:SparseArrays] = @benchmarkable begin
    C = similar(A, (N, M))
    SparseArrays._spmul!(C, A, B, 1, 0) 
  end setup = begin A = rand($N, $K); B = sprand($K, $M, 0.05) end
suite[:ThisApproach] = @benchmarkable begin
    # we could also just do A*B, but we want to stay consistent with the above
    C = similar(A, (N, M))
    mul!(C, A, B, 1, 0)
  end setup = begin A = rand($N, $K); B = sprand($K, $M, 0.05) end

suite[:SparseArrays_noalloc] = @benchmarkable begin
    SparseArrays._spmul!(C, A, B, 1, 0) 
  end setup = begin A = rand($N, $K); B = sprand($K, $M, 0.05); C = similar(A, ($N, $M)) end
suite[:ThisApproach_noalloc] = @benchmarkable begin
    # we could also just do A*B, but we want to stay consistent with the above
    mul!(C, A, B, 1, 0)
  end setup = begin A = rand($N, $K); B = sprand($K, $M, 0.05); C = similar(A, ($N, $M)) end

# These turn out to be extremely slow...
# suite[:SparseMatricesCSR_noalloc] = @benchmarkable begin
#     # we could also just do A*B, but we want to stay consistent with the above
#     mul!(C, A, B, 1, 0)
#   end setup = begin A = rand(N, K); B = sparsecsr(findnz(sprand(K, M, 0.05))..., K, M); C = similar(A, (N, M)) end
# suite[:SparseMatricesCSR] = @benchmarkable begin
#     # we could also just do A*B, but we want to stay consistent with the above
#     C = similar(A, (N, M))
#     mul!(C, A, B, 1, 0)
#   end setup = begin A = rand(N, K); B = sparsecsr(findnz(sprand(K, M, 0.05))..., K, M);  end

tune!(suite)
res = run(suite)
fig = plot(res)
savefig(fig, "benchmark/benchmark.svg")
savefig(fig, "benchmark/benchmark.png")



# scaling experiments
M_values = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
suite2 = BenchmarkGroup()
suite2[:SparseArrays] = BenchmarkGroup()
for M in M_values
  suite2[:SparseArrays][M] = @benchmarkable begin
    SparseArrays._spmul!(C, A, B, 1, 0) 
  end setup = begin A = rand($N, $K); B = sprand($K, $M, 0.05); C = similar(A, ($N, $M)) end
end
suite2[:DenseSparseMul] = BenchmarkGroup()
for M in M_values
  suite2[:DenseSparseMul][M] = @benchmarkable begin
    mul!(C, A, B, 1, 0)
  end setup = begin A = rand($N, $K); B = sprand($K, $M, 0.05); C = similar(A, ($N, $M)) end
end
# tune!(suite2)
res2 = run(suite2)

fig = plot(; xaxis=:log10, yaxis=:log2, 
           title="3x Speedup over StaticArrays.jl", xlabel="M", ylabel="time [ms]",
           minorgrid=true, xticks=10 .^[2:5...], yticks=2. .^[0:2:12...], dpi=300)
plot!(fig, M_values, [median(res2[:SparseArrays][M]).time/1e6 for M in M_values], label="StaticArrays.jl", marker=:x)
plot!(fig, M_values, [median(res2[:DenseSparseMul][M]).time/1e6 for M in M_values], label="DenseSparseMul.jl", marker=:x)
savefig(fig, "benchmark/scaling.png")
savefig(fig, "benchmark/scaling.svg")

