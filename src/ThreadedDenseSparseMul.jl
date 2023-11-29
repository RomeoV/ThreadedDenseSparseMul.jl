module ThreadedDenseSparseMul

import SparseArrays
import SparseArrays: SparseMatrixCSC, SparseVector, nonzeroinds, nonzeros, findnz
import Polyester: @batch

"""
    fastdensesparse!(C, A, B, α, β)

BLAS like interface, computing `C .= β*C + α*A*B`, but accelerated with multi-threading using `Polyester.jl`.
"""
function fastdensesparse!(C::Matrix{T}, A::Matrix{T}, B::SparseMatrixCSC{T}, α::Number, β::Number) where T
    for j in axes(B, 2)
        C[:, j] .*= β
        C[:, j] .+= A * (α.*B[:, j])
    end
    return C
end
function fastdensesparse_threaded!(C::Matrix{T}, A::Matrix{T}, B::SparseMatrixCSC{T}, α::Number, β::Number) where T
    @batch for j in axes(B, 2)
        C[:, j] .*= β
        C[:, j] .+= A * (α.*B[:, j])
    end
    return C
end

const VecOrView{T} = Union{Vector{T}, SubArray{T, 1, Matrix{T}}}
# this one is slightly slower than `fastdensesparse_outer!`, probably because of extra allocations.
function _fastdensesparse_outer!(C::Matrix{T}, A::VecOrView{T}, b::SparseVector{T}, α::Number, β::Number) where T
    for (j, X_val) in zip(findnz(b)...)
        C[:, j] .*= β
        C[:, j] .+=  (α*X_val) .* A  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
    end
    return C
end

function fastdensesparse_outer!(C::Matrix{T}, a::VecOrView{T}, b::SparseVector{T}, α::Number, β::Number) where T
    C[:, nonzeroinds(b)] .+=  a * nonzeros(b)'
    return C
end

function fastdensesparse_outer_threaded!(C::Matrix{T}, a::VecOrView{T}, b::SparseVector{T}, α::Number, β::Number) where T
    inds = nonzeroinds(b)
    nzs = nonzeros(b)
    @batch for i in axes(nzs, 1)
        C[:, inds[i]] .*=  β
        C[:, inds[i]] .+=  (α.*nzs[i]).*a
    end
    return C
end



end # module ThreadedDenseSparseMul
