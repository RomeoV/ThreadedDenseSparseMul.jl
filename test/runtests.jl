using ThreadedDenseSparseMul
using Test
import SparseArrays: mul!, sprand
import Profile

# Helper function to run common test logic
function run_common_tests(method!, buf::AbstractMatrix{T}, lhs, rhs, α, β, baseline) where {T <: Number}
    method!(buf, lhs, rhs, α, β)
    @test buf ≈ baseline rtol=sqrt(eps(real(T)))
    @test !any(isnan, buf)
    
    # Test with negative α
    method!(buf, lhs, rhs, -α, β)
    method!(buf, lhs, rhs, α, β)
    @test buf ≈ baseline rtol=sqrt(eps(real(T)))
    @test !any(isnan, buf)
end

@testset "ThreadedDenseSparseMul Tests" begin
    @test ThreadedDenseSparseMul.get_num_threads() == Threads.nthreads()
    @testset "Override mul! ?" for override_mul! in [false, true]
        override_mul! && ThreadedDenseSparseMul.override_mul!()
        @testset "nthreads" for nthreads in [1, Threads.nthreads()]
            ThreadedDenseSparseMul.set_num_threads(nthreads)
            include("test_densesparsemul.jl")
            include("test_densesparseouter.jl")
        end
    end
end
