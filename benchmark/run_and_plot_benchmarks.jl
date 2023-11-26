raw"""
    Summary of results:

For matrices (NxK) and (KxM) we fix N=1_000 and K=2_000, and vary N.
For all N we see a speed up, up to approximately 3x for N=30_000.

We compare our approach against the SparseArary.sjl baseline, and against MKLSparse.jl.
However, since MKLSparse doesn't offer dense $\cdot$ sparse (but only sparse $\cdot$ dense) there is additional allocation overhead.
We also compare this aginst SparseArrays where we do the same.
"""

using SparseArrays # SparseMatricesCSR, ThreadedSparseCSR
using Plots, StatsPlots, BenchmarkPlots, BenchmarkTools
using ThreadedDenseSparseMul  # note that this overwrites the regular mul!
import SparseArrays: mul!, findnz
import Profile
import ThreadPinning: pinthreads
using MKLSparse
import LinearAlgebra: wrapper_char, MulAddMul
pinthreads(:cores)
include("util.jl")

N, K, M = 1_000, 2_000, 30_000;

# Since we load multiple packages that overload `mul!` we first check that our workloads actually only use the modules they're meant to use!

# Check MKL Sparse workload
let
    A = rand(N, K); B = sprand(K, M, 0.05); C = similar(A, (N, M))
    C_ = similar(C, reverse(size(C))); A_ = Matrix(A')

    # Check MKLSparse workload
    Profile.clear(); Profile.@profile mul!(C_, B', A_, true, false);
    @assert !check_module_used("Polyester", Profile.fetch())
    @assert !check_module_used("ThreadedDenseSparseMul", Profile.fetch())
    @assert  check_module_used("MKLSparse", Profile.fetch())

    # Check "our" workload
    Profile.clear(); Profile.@profile mul!(C, A, B, true, false);
    @assert  check_module_used("Polyester", Profile.fetch())
    @assert  check_module_used("ThreadedDenseSparseMul", Profile.fetch())
    @assert !check_module_used("MKLSparse", Profile.fetch())
    @assert  check_module_used("SparseArrays", Profile.fetch())  # for matrix times vector

    # Check SparseArrays workload
    Profile.clear(); Profile.@profile SparseArrays._spmul!(C, A, B, true, false);
    @assert !check_module_used("Polyester", Profile.fetch())
    @assert !check_module_used("ThreadedDenseSparseMul", Profile.fetch())
    @assert !check_module_used("MKLSparse", Profile.fetch())
    @assert  check_module_used("SparseArrays", Profile.fetch())

    # Check SparseArraysT workload
    Profile.clear(); Profile.@profile SparseArrays.spdensemul!(C_, wrapper_char(B'), wrapper_char(A), B, A_, MulAddMul(true, false));
    @assert !check_module_used("Polyester", Profile.fetch())
    @assert !check_module_used("ThreadedDenseSparseMul", Profile.fetch())
    @assert !check_module_used("MKLSparse", Profile.fetch())
    @assert  check_module_used("SparseArrays", Profile.fetch())
end

## Now we're ready to set up the benchmarks
suite = BenchmarkGroup()

# scaling experiments
M_values = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
suite = BenchmarkGroup()

suite[:SparseArrays] = BenchmarkGroup()
for M in M_values
  suite[:SparseArrays][M] = @benchmarkable begin
    SparseArrays._spmul!(C, A, B, 1, 0) 
  end setup = begin A = rand($N, $K); B = sprand($K, $M, 0.05); C = similar(A, ($N, $M)) end
end

suite[:SparseArraysT] = BenchmarkGroup()
for M in M_values
  suite[:SparseArraysT][M] = @benchmarkable begin
    C_ = similar(C, reverse(size(C)))
    A_ = Matrix(A')
    SparseArrays.spdensemul!(C_, wrapper_char(B'), wrapper_char(A), B, A_, MulAddMul(true, false))
    C .= C_'
  end setup = begin A = rand($N, $K); B = sprand($K, $M, 0.05); C = similar(A, ($N, $M)) end
end

suite[:DenseSparseMul] = BenchmarkGroup()
for M in M_values
  suite[:DenseSparseMul][M] = @benchmarkable begin
    mul!(C, A, B, 1, 0)
  end setup = begin A = rand($N, $K); B = sprand($K, $M, 0.05); C = similar(A, ($N, $M)) end
end

suite[:MKLSparse] = BenchmarkGroup()
for M in M_values
  suite[:MKLSparse][M] = @benchmarkable begin
    C_ = similar(C, reverse(size(C)))
    A_ = Matrix(A')
    mul!(C_, B', A_, 1, 0)
    C .= C_'
  end setup = begin A = rand($N, $K); B = sprand($K, $M, 0.05); C = similar(A, ($N, $M)) end
end

suite[:MKLSparse_no_alloc] = BenchmarkGroup()
for M in M_values
  suite[:MKLSparse_no_alloc][M] = @benchmarkable begin
    mul!(C_, B', A_, 1, 0)
  end setup = begin
      A = rand($N, $K); B = sprand($K, $M, 0.05); C = similar(A, ($N, $M))
      C_ = similar(C, reverse(size(C))); A_ = Matrix(A')
  end
end
# tune!(suite)
res = run(suite)

fig = plot(; xaxis=:log10, yaxis=:log2, 
           title="2x Speedup over SparseArrays.jl", xlabel="M", ylabel="time [ms]",
           minorgrid=true, xticks=10 .^[2:5...], yticks=2. .^[0:2:12...], dpi=300,
           legend=:topleft)
plot!(fig, M_values, [median(res[:SparseArrays][M]  ).time/1e6 for M in M_values], label="SparseArrays.jl", marker=:x)
plot!(fig, M_values, [median(res[:SparseArraysT][M] ).time/1e6 for M in M_values], label="SparseArraysT.jl", marker=:x)
plot!(fig, M_values, [median(res[:MKLSparse][M]     ).time/1e6 for M in M_values], label="MKLSparse.jl", marker=:x)
plot!(fig, M_values, [median(res[:MKLSparse_no_alloc][M]).time/1e6 for M in M_values], label="MKLSparse.jl (no alloc)", marker=:x)
plot!(fig, M_values, [median(res[:DenseSparseMul][M]).time/1e6 for M in M_values], label="DenseSparseMul.jl", marker=:x)
savefig(fig, "benchmark/scaling.png")
savefig(fig, "benchmark/scaling.svg")

