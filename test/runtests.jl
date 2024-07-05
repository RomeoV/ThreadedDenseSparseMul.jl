using ThreadedDenseSparseMul
using Test
import SparseArrays: mul!, sprand
import Profile

# Helper function to run common test logic
function run_common_tests(method!, buf::AbstractMatrix{T}, lhs, rhs, α, β, baseline) where {T <: Real}
    method!(buf, lhs, rhs, α, β)
    @test buf ≈ baseline rtol=sqrt(eps(T))
    @test !any(isnan, buf)
    
    # Test with negative α
    method!(buf, lhs, rhs, -α, β)
    method!(buf, lhs, rhs, α, β)
    @test buf ≈ baseline rtol=sqrt(eps(T))
    @test !any(isnan, buf)
end

@testset "ThreadedDenseSparseMul Tests" begin
    @test ThreadedDenseSparseMul.get_num_threads() == Threads.nthreads()
    @testset "Dense-Sparse Multiplication" begin
        @testset "$T type" for T in [Float64, Float32]
            @testset "$method! implementation" for method! in [fastdensesparsemul!, fastdensesparsemul_threaded!]
                @testset "Trial $trial" for trial in 1:10
                    lhs = rand(T, 50, 100)
                    rhs = sprand(T, 100, 1_000, 0.1)
                    baseline = lhs * rhs
                    
                    buf = similar(baseline) .* false  # fill buffer with zeros. Carefull with NaNs, see https://discourse.julialang.org/t/occasionally-nans-when-using-similar/48224/12

                    # Test basic multiplication
                    run_common_tests(method!, buf, lhs, rhs, 1, 0, baseline)
                    
                    # Test @view interface and β ≠ 0
                    inds = rand(axes(lhs, 1), size(lhs, 1) ÷ 3)
                    baseline[inds, :] .+= 2.5 * @view(lhs[inds, :]) * rhs
                    
                    run_common_tests(method!, @view(buf[inds, :]), @view(lhs[inds, :]), rhs,  2.5, 1, @view(baseline[inds, :]))
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
                    k = rand(axes(rhs, 1))
                    
                    baseline = lhs[:, k:k] * rhs[k:k, :]
                    buf = similar(baseline) .* false
                    
                    # Test basic outer product multiplication
                    run_common_tests(method!, buf, @view(lhs[:, k]), rhs[k, :], 1, 0, baseline)
                    
                    # Test with β ≠ 0
                    baseline .+= 2.5 * lhs[:, (k+1):(k+1)] * rhs[(k+1):(k+1), :]
                    run_common_tests(method!, buf, lhs[:, k+1], rhs[k+1, :], 2.5, 1, baseline)
                end
            end
        end
    end
end
