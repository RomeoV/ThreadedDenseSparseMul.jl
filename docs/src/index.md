# ThreadedDenseSparseMul.jl

ThreadedDenseSparseMul.jl is a Julia package that provides a threaded implementation of dense-sparse matrix multiplication, built on top of `Polyester.jl`.

## Installation

You can install ThreadedDenseSparseMul.jl using Julia's package manager:

```julia
using Pkg
Pkg.add("ThreadedDenseSparseMul")
```

## Usage

To use ThreadedDenseSparseMul.jl, simply install and import the package, and launch Julia with some threads (e.g., `julia --threads=auto`). Then, you can use any of the following accelerated functions:

```julia
using ThreadedDenseSparseMul
using SparseArrays

A = rand(1_000, 2_000)
B = sprand(2_000, 30_000, 0.05)
buf = similar(A, size(A,1), size(B,2))  # prealloc

fastdensesparsemul!(buf, A, B, 1, 0)
fastdensesparsemul_threaded!(buf, A, B, 1, 0)
fastdensesparsemul_outer!(buf, @view(A[:, 1]), B[1,:], 1, 0)
fastdensesparsemul_outer_threaded!(buf, @view(A[:, 1]), B[1,:], 1, 0)
```

The interface is adapted from the 5-parameter definition used by `mul!` and BLAS.

## API Reference

- `fastdensesparsemul!(C, A, B, α, β)`: Computes `C = β*C + α*A*B`
- `fastdensesparsemul_threaded!(C, A, B, α, β)`: Threaded version of `fastdensesparsemul!`
- `fastdensesparsemul_outer!(C, a, b, α, β)`: Computes `C = β*C + α*a*b'`
- `fastdensesparsemul_outer_threaded!(C, a, b, α, β)`: Threaded version of `fastdensesparsemul_outer!`

## Rationale

This package addresses the need for fast computation of `C ← C - D × S`, where `D` and `S` are dense and sparse matrices, respectively. It differs from existing solutions in the following ways:

- SparseArrays.jl doesn't support threaded multiplication.
- IntelMKL.jl doesn't support dense × sparsecsc multiplication directly.
- ThreadedSparseCSR.jl and ThreadedSparseArrays.jl support only sparse × dense multiplication.

## Implementation

The core implementation is quite simple, leveraging `Polyester.jl` for threading:

```julia
function fastdensesparsemul_threaded!(C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC, α::Number, β::Number)
    @batch for j in axes(B, 2)
        C[:, j] .*= β
        C[:, j] .+= A * (α.*B[:, j])
    end
    return C
end
```

## Performance

ThreadedDenseSparseMul.jl shows significant performance improvements over base Julia implementations, especially for large matrices. Benchmark results comparing against SparseArrays.jl and MKLSparse.jl are available in the repository.

## Contributing

Contributions to ThreadedDenseSparseMul.jl are welcome! Please feel free to submit issues or pull requests on the GitHub repository.

## License

ThreadedDenseSparseMul.jl is licensed under the MIT License. See the LICENSE file in the package repository for more details.
