using SparseArrays # SparseMatricesCSR, ThreadedSparseCSR
using Plots, StatsPlots, BenchmarkPlots, BenchmarkTools
using ThreadedDenseSparseMul  # note that this overwrites the regular mul!
import Profile
import ThreadPinning: pinthreads
import Random: shuffle
pinthreads(:cores)
# include("util.jl")

N, K, M = 1_000, 2_000, 30_000;

lhs = rand(N, K);
rhs = sprand(K, M, 0.05);
buf = rand(N, M);

suite = BenchmarkGroup()
suite[:Float64] = BenchmarkGroup()
suite[:Float64][:fastdensesparse!] = @benchmarkable ThreadedDenseSparseMul.fastdensesparse!(buf, lhs, rhs, 2., 0)
suite[:Float64][:fastdensesparse_threaded!] = @benchmarkable ThreadedDenseSparseMul.fastdensesparse_threaded!(buf, lhs, rhs, 2., 0)


tune!(suite)
res1 = run(suite)
show(res1)

suite = BenchmarkGroup()
suite[:Float64] = BenchmarkGroup()
suite[:Float64][:_fastdensesparse_outer!] = @benchmarkable ThreadedDenseSparseMul._fastdensesparse_outer!(buf, @view(lhs[:, 10]), rhs[10, :], 2., 0)
suite[:Float64][:fastdensesparse_outer!] = @benchmarkable ThreadedDenseSparseMul.fastdensesparse_outer!(buf, @view(lhs[:, 10]), rhs[10, :], 2., 0)
suite[:Float64][:base] = @benchmarkable buf .= @view(lhs[:, 10])*(2. .*rhs[10, :])'
suite[:Float64][:fastdensesparse_outer_threaded!] = @benchmarkable ThreadedDenseSparseMul.fastdensesparse_outer_threaded!(buf, @view(lhs[:, 10]), rhs[10, :], 2., 0)
tune!(suite)
res2 = run(suite)
show(res2);


# inside multithreading
ks = shuffle(1:K)[1:100]
bufs = [copy(buf) for _ in 1:Threads.nthreads()]
suite = BenchmarkGroup()
suite[:Float64] = BenchmarkGroup()
suite[:Float64][:_fastdensesparse_outer!] = @benchmarkable begin
    buf = bufs[Threads.threadid()]
    Threads.@threads :static for k in ks
        ThreadedDenseSparseMul._fastdensesparse_outer!(buf, @view(lhs[:, k]), rhs[k, :], 2., 0)
    end
end
suite[:Float64][:fastdensesparse_outer!] = @benchmarkable begin
    buf = bufs[Threads.threadid()]
    Threads.@threads :static for k in ks
        ThreadedDenseSparseMul.fastdensesparse_outer!(buf, @view(lhs[:, k]), rhs[k, :], 2., 0)
    end
end
suite[:Float64][:fastdensesparse_outer_threaded!] = @benchmarkable begin
    buf = bufs[Threads.threadid()]
    Threads.@threads :static for k in ks
        ThreadedDenseSparseMul.fastdensesparse_outer_threaded!(buf, @view(lhs[:, k]), rhs[k, :], 2., 0)
    end
end
tune!(suite)
res3 = run(suite)
show(res3);
