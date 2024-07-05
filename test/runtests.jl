using ThreadedDenseSparseMul
using Test
import SparseArrays: mul!, sprand
import Profile

# Helper function to run common test logic
function run_common_tests(method!, lhs, rhs, baseline, buf, α, β)
    method!(buf, lhs, rhs, α, β)
    @test buf ≈ baseline rtol=sqrt(eps(eltype(baseline)))
    @test !any(isnan, buf)
    
    # Test with negative α
    method!(buf, lhs, rhs, -α, β)
    method!(buf, lhs, rhs, α, β)
    @test buf ≈ baseline rtol=sqrt(eps(eltype(baseline)))
    @test !any(isnan, buf)
end

@testset "ThreadedDenseSparseMul Tests" begin
    @testset "Dense-Sparse Multiplication" begin
        @testset "$T type" for T in [Float64, Float32]
            @testset "$method! implementation" for method! in [fastdensesparsemul!, fastdensesparsemul_threaded!]
                @testset "Trial $trial" for trial in 1:10
                    lhs = rand(T, 50, 100)
                    rhs = sprand(T, 100, 1_000, 0.1)
                    baseline = lhs * rhs
                    
                    # Initialize buffer (avoiding potential NaN issues)
                    buf = similar(baseline)
                    buf .= zero(T)
                    
                    # Test basic multiplication
                    run_common_tests(method!, lhs, rhs, baseline, buf, 1, 0)
                    
                    # Test @view interface and β ≠ 0
                    inds = collect(3:5:50)
                    baseline[inds, :] .+= 2.5 * @view(lhs[inds, :]) * rhs
                    
                    run_common_tests(method!, @view(lhs[inds, :]), rhs, @view(baseline[inds, :]), @view(buf[inds, :]), 2.5, 1)
                end
            end
        end
    end

    @testset "Outer Product Multiplication" begin
        @testset "$T type" for T in [Float64, Float32]
            @testset "$method! implementation" for method! in [fastdensesparsemul_outer!, fastdensesparsemul_outer_threaded!]
                @testset "Trial $trial" for trial in 1:10
                    lhs = rand(T, 50, 100)
                    rhs = sprand(T, 100, 1_000, 0.1)
                    k = rand(1:size(rhs, 1))
                    
                    baseline = lhs[:, k:k] * rhs[k:k, :]
                    buf = zeros(T, size(baseline))
                    
                    # Test basic outer product multiplication
                    run_common_tests(method!, @view(lhs[:, k]), rhs[k, :], baseline, buf, 1, 0)
                    
                    # Test with β ≠ 0
                    baseline .+= 2.5 * lhs[:, (k+1):(k+1)] * rhs[(k+1):(k+1), :]
                    run_common_tests(method!, lhs[:, k+1], rhs[k+1, :], baseline, buf, 2.5, 1)
                end
            end
        end
    end
end
