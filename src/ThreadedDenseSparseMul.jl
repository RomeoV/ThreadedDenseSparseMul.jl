module ThreadedDenseSparseMul

import SparseArrays
import SparseArrays: SparseMatrixCSC, mul!
import Polyester: @batch

function SparseArrays.mul!(C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC, α::Number, β::Number)
    @batch for j in axes(B, 2)
        C[:, j] .*= β
        C[:, j] .+= A * (α.*B[:, j])
    end
    return C
end

end # module ThreadedDenseSparseMul
