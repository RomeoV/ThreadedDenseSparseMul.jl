# ThreadedDenseSparseMul.jl
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://romeov.github.io/ThreadedDenseSparseMul.jl/dev/)
[![Build Status](https://github.com/JuliaCI/BenchmarkTools.jl/workflows/CI/badge.svg)](https://github.com/JuliaCI/BenchmarkTools.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/RomeoV/ProbabilisticParameterEstimators.jl/graph/badge.svg?token=5J82UXPL8I)](https://codecov.io/gh/RomeoV/ThreadedDenseSparseMul.jl)
[![Aqua](https://img.shields.io/badge/Tested_with-Aqua-turquoise)](https://github.com/JuliaTesting/Aqua.jl)

> Threaded implementation of dense-sparse matrix multiplication, built on top of `Polyester.jl`.

## Usage
Just install and import this package, and launch Julia with some threads `e.g. julia --threads=auto`! Then e.g. any of these will be accelerated:
```julia
A = rand(1_000, 2_000); B = sprand(2_000, 30_000, 0.05); buf = similar(size(A,1), size(B,2))  # prealloc
fastdensesparsemul!(buf, A, B, 1, 0)
fastdensesparsemul_threaded!(buf, A, B, 1, 0)
fastdensesparsemul_outer!(buf, @view(A[:, 1]), B[1,:], 1, 0)
fastdensesparsemul_outer_threaded!(buf, @view(A[:, 1]), B[1,:], 1, 0)
```

The interface is adapted from the 5-parameter definition used by `mul!` and also BLAS.
Previously we tried to overload the `mul!` operator, which comes with some nice syntactic convenience, but it lead to some downstream problems with package precompilation, and checking whether our implementation is actually used.

## Rationale
I want to do $C \leftarrow C - D \times S$ fast, where $D$ and $S$ are dense and sparse matrices, respectively.
Notice how this is different from $C \leftarrow C - S \times D$, i.e. dense $\times$ sparse vs sparse $\times$ dense.
In particular:
- The SparseArrays.jl package doesn't support threaded multiplication.
- The IntelMKL.jl package doesn't seem to support dense $\times$ sparsecsc multiplication, although one can get similar performance using that package and transposing appropriately. It also comes with possible licensing issues and is vendor-specific.
- The ThreadedSparseCSR.jl package also just supports sparsecsr $\times$ dense.
- The ThreadedSparseArrays.jl package also just supports ThreadedSparseMatrixCSC $\times$ dense, and also doesn't install for me currently.

I haven't found an implementation for that, so made one myself. In fact, the package `Polyester.jl` makes this super easy, the entire code is basically
```julia
import SparseArrays: SparseMatrixCSC, mul!; import SparseArrays
import Polyester: @batch

function fastdensesparsemul_threaded!(C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC, α::Number, β::Number)
    @batch for j in axes(B, 2)
        C[:, j] .*= β
        C[:, j] .+= A * (α.*B[:, j])
    end
    return C
end
```

Notice that this approach doesn't make sense for matrix-vector multiplication (the loop would just have one element), so that case is not considered in this package, however it does make sense for outer producs.


### Note on column-major and CSC vs CSR
I haven't found much literature on the choice of CSC vs CSR storage specifically of the context of dense $\cdot$ sparse multiplication with column-major dense storage.
However, as we can see in the code snippet above, the CSC format seems to be reasonably sensible for column-major dense storage.
To compute any given column in $C_{(:,j)}$ of $C$ we are essentially computing a weighted sum of columns in $A$, i.e. $C_{(:,j)} = \sum_k \lambda_k \cdot A_{(:,k)}$ which should be very cache efficient and SSE-able.

## Benchmarking
For matrices $(N\times K)$ and $(K\times M)$ we fix $N=1'000$ and $K=2'000$ and vary M.
Here's are the benchmark results, comparing against SparseArrays.jl, which ships with Julia but is single-threaded:

![scaling benchmark](/benchmark/scaling.png)

For all M we see a speed up over `_spmul!` from the SparseArrays package of up to ~2x for M in [300, 30_000].
We also compare against `MKLSparse.jl`. However, since MKLSparse only supports `dense x sparse` we first need to allocate spare buffers and transpose the dense matrix (these allocations are not measured in the `no_transpose` variant), and then computing essentially $(B^T A^T)^T$.
The result is much slower, likely due to the fact that the dense matrix is column-major.
We also compare against SparseArrays.jl doing the same, where we also see poor performance.
