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
