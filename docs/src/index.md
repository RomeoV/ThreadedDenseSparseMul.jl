# ThreadedDenseSparseMul.jl

`ThreadedDenseSparseMul.jl` is a Julia package that provides a threaded implementation of dense-sparse matrix multiplication, built on top of `Polyester.jl`.

This package addresses the need for fast computation of `C ← C + D × S`, where `D` and `S` are dense and sparse matrices, respectively. It differs from existing solutions in the following ways:

- SparseArrays.jl doesn't support threaded multiplication.
- MKLSparse.jl doesn't support dense × sparsecsc multiplication directly.
- ThreadedSparseCSR.jl and ThreadedSparseArrays.jl support only sparse × dense multiplication.

ThreadedDenseSparseMul.jl shows significant performance improvements over base Julia implementations, especially for large matrices.

## Performance
#### [`fastdensesparsemul_threaded!`](@ref) outperforms `MKLSparse` by 2x:

![`fastdensesparsemul!` outperforms MKLSparse by 2x.](assets/main.svg)

#### [`fastdensesparsemul_outer_threaded!`](@ref) outperforms `SparseArrays` by 4x:

![`fastdensesparsemulmul!` outperforms SparseArrays for outer product by 4x.](assets/main_outer.svg)

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
```@autodocs
Modules = [ThreadedDenseSparseMul]
```

## Implementation

The core implementation is quite simple, leveraging `Polyester.jl` for threading. The result is simply something similar to

```julia
function fastdensesparsemul_threaded!(C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC, α::Number, β::Number)
    @batch for j in axes(B, 2)
        C[:, j] .*= β
        C[:, j] .+= A * (α.*B[:, j])
    end
    return C
end
```

## Contributing

Contributions to ThreadedDenseSparseMul.jl are welcome! Please feel free to submit issues or pull requests on the GitHub repository.

## License

ThreadedDenseSparseMul.jl is licensed under the MIT License. See the LICENSE file in the package repository for more details.
