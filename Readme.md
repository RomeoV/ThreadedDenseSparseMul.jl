### Rationale
- The SparseArrays.jl package doesn't support threaded multiplication.
- The IntelMKL.jl package doesn't seem to support dense*sparsecsc multiplication, although one can get similar performance using that package and transposing appropriately.
- The ThreadedSparseCSR.jl package also just supports sparsecsr*dense.
- The ThreadedSparseArrays.jl package also just supports ThreadedSparseMatrixCSC*dense, and also doesn't install for me currently.

For my application I would like to compute dense*sparse.
I haven't found an implementation for that, so made one myself. Actually, the package `Polyester.jl` makes this super easy, the entire code is basically
```julia
import SparseArrays: SparseMatrixCSC, mul!; import SparseArrays
import Polyester: @batch

function SparseArrays.mul!(C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC, α::Number, β::Number)
    @batch for j in axes(B, 2)
        C[:, j] .*= β
        C[:, j] .+= A * (α.*B[:, j])
    end
    return C
end
```

Julia will automatically use this 5-parameter definition to generate `mul!(C, A, B)` and calls like `C .+= A*B` and so forth.


#### Note on column-major and CSC vs CSR
I haven't found much literature on the choice of CSC vs CSR storage specifically of the context of dense*sparse multiplication with column major storage for the dense matrix.
However, as we can see in the code snippet above, the CSC format seems to be reasonably sensible for column-major (`C[:, j]`) and CSC (`B[:, j]`).

#### Benchmarking
For matrices (NxK) and (KxM) we fix N=1_000 and K=2_000, and vary N.
<<<<<<< HEAD
For all N we see a speed up over `_spmul!` from the StaticArrays package of up to ~3x for N in [300, 30_000], and ~2x otherwise.

Here's the benchmark:
![scaling benchmark](/benchmark/scaling.png)
=======
For all N we see a speed up over `_spmul!` from the StaticArrays package of up to ~3x for N=30_000.
>>>>>>> 3efbbfb (Add Readme)
