using ThreadedDenseSparseMul, SparseArrays
using Test
import SparseArrays: _spmul!, mul!

SparseArrays._spmul!(C::Matrix{Float64}, X::Matrix{Float64}, A::SparseMatrixCSC{Float64, Int64}, α::Number, β::Number) = @assert false

@testset "Compare with equivalent dense mul" begin
  @testset for trial in 1:10
    lhs = rand(100, 120);
    rhs = sprand(120, 140, 0.1);

    baseline = lhs * Matrix(rhs);
    @test lhs * rhs ≈ baseline

    buf = similar(baseline)
    buf .= lhs * rhs
    @test buf ≈ baseline

    buf .= 0.
    buf .+= lhs * rhs
    @test buf ≈ baseline

    mul!(buf, lhs, rhs, 1, 0)
    @test buf ≈ baseline

    # make sure I've overwritten the regular spmul correctly...
    @test_throws AssertionError _spmul!(buf, lhs, rhs, 1., 1.)
  end
end
