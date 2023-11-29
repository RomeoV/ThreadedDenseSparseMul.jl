module ThreadedDenseSparseMul

import SparseArrays
import SparseArrays: SparseMatrixCSC, SparseVector, nonzeroinds, nonzeros, findnz
import Polyester: @batch

export fastdensesparse!, fastdensesparse_threaded!
export fastdensesparse_outer!, fastdensesparse_outer_threaded!

const VecOrView{T} = Union{Vector{T}, SubArray{T, 1, Matrix{T}}}
const MatOrView{T} = Union{Matrix{T}, SubArray{T, 2, Matrix{T}}}

"""
    fastdensesparse!(C, A, B, α, β)

BLAS like interface, computing `C .= β*C + α*A*B`, but way faster than Base would.

Also see `fastdensesparse_threaded!` for a multi-threaded version using `Polyester.jl`.
"""
function fastdensesparse!(C::MatOrView{T}, A::MatOrView{T}, B::SparseMatrixCSC{T}, α::Number, β::Number) where T
    for j in axes(B, 2)
        C[:, j] .*= β
        C[:, j] .+= A * (α.*B[:, j])
    end
    return C
end

"""
    fastdensesparse!(C, A, B, α, β)

Threaded, BLAS like interface, computing `C .= β*C + α*A*B`, but way faster than Base would.
Also see `fastdensesparse!` for a single-threaded version.
"""
function fastdensesparse_threaded!(C::MatOrView{T}, A::MatOrView{T}, B::SparseMatrixCSC{T}, α::Number, β::Number) where T
    @batch for j in axes(B, 2)
        C[:, j] .*= β
        C[:, j] .+= A * (α.*B[:, j])
    end
    return C
end

"""
    fastdensesparse_outer!(C, a, b, α, β)

Fast outer product when computing `C .= β*C + α * a*b'`, but way faster than Base would.
- `a` is a dense vector (or view), `b` is a sparse vector, `C` is a dense matrix (or view).
Also see `fastdensesparse_outer_threaded!` for a multi-threaded version using `Polyester.jl`.
"""
function fastdensesparse_outer!(C::MatOrView{T}, a::VecOrView{T}, b::SparseVector{T}, α::Number, β::Number) where T
    C[:, nonzeroinds(b)] .+=  a * nonzeros(b)'
    return C
end

"""
    fastdensesparse_outer_threaded!(C, a, b, α, β)

Threaded, fast outer product when computing `C .= β*C + α * a*b'`, but way faster than Base would, using `Polyester.jl`.
- `a` is a dense vector (or view), `b` is a sparse vector, `C` is a dense matrix (or view).

Also see `fastdensesparse_outer!` for a single-threaded version.
"""
function fastdensesparse_outer_threaded!(C::MatOrView{T}, a::VecOrView{T}, b::SparseVector{T}, α::Number, β::Number) where T
    inds = nonzeroinds(b)
    nzs = nonzeros(b)
    @batch for i in axes(nzs, 1)
        C[:, inds[i]] .*=  β
        C[:, inds[i]] .+=  (α.*nzs[i]).*a
    end
    return C
end

end # module ThreadedDenseSparseMul
