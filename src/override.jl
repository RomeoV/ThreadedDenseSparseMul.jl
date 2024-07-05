import LinearAlgebra: mul!, Adjoint

function override_mul!(threaded = true)
    if threaded
        @eval function  mul!(C::MatOrView{T}, A::MatOrView{T}, B::SparseMatrixCSC{T}, α::Number, β::Number) where T
            fastdensesparsemul_threaded!(C, A, B, α, β)
        end
        @eval function mul!(C::MatOrView{T}, a::VecOrView{T}, b::Adjoint{<:SparseVector{T}}, α::Number, β::Number) where T
            fastdensesparsemul_outer_threaded!(C, a, b', α, β)
        end
    else
        @eval function  mul!(C::MatOrView{T}, A::MatOrView{T}, B::SparseMatrixCSC{T}, α::Number, β::Number) where T
            fastdensesparsemul!(C, A, B, α, β)
        end
        @eval function mul!(C::MatOrView{T}, a::VecOrView{T}, b::Adjoint{<:SparseVector{T}}, α::Number, β::Number) where T
            fastdensesparsemul_outer!(C, a, b', α, β)
        end
    end
end
