import LinearAlgebra: mul!, Adjoint
import Base: (*)

function override_mul!(threaded = true)
    if threaded
        @eval function  mul!(C::MatOrView{T}, A::MatOrView{T}, B::SparseMatrixCSC{T}, α::Number, β::Number) where T
            fastdensesparsemul_threaded!(C, A, B, α, β)
        end
        @eval function (*)(a::VecOrView{T}, b::Adjoint{T, <:SparseVector{T}}) where T
            res = similar(a, length(a), length(b))
            fastdensesparsemul_outer_threaded!(res, a, b', true, false)
            res
        end
    else
        @eval function  mul!(C::MatOrView{T}, A::MatOrView{T}, B::SparseMatrixCSC{T}, α::Number, β::Number) where T
            fastdensesparsemul!(C, A, B, α, β)
        end
        @eval function (*)(a::VecOrView{T}, b::Adjoint{T, <:SparseVector{T}}) where T
            res = similar(a, length(a), length(b))
            fastdensesparsemul_outer!(res, a, b', true, false)
            res
        end
    end
end
