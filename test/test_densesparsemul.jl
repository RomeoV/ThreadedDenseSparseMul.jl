@testset "Dense-Sparse Multiplication" begin
    @testset "$T type" for T in [Float64, Float32, Float16, Complex{Float32}]
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
