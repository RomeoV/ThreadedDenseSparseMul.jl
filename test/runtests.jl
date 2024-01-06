using ThreadedDenseSparseMul
using Test
import SparseArrays: mul!, sprand
import Profile


@testset "Compare with equivalent dense mul" begin
  @testset "fastdensesparsemul" begin
    @testset for T in [Float64, Float32]
      @testset for method! in [fastdensesparsemul!, fastdensesparsemul_threaded!]
        @testset for trial in 1:10
          lhs = rand(T, 50, 100);
          rhs = sprand(T, 100, 1_000, 0.1);

          baseline = lhs * rhs;

          # buf = 0*similar(baseline)  # <- BUG!! See https://discourse.julialang.org/t/occasionally-nans-when-using-similar/48224/12
          # use instead
          # buf = zeros(T, size(baseline))
          # buf = similar(baseline); buf .= zero(T)
          # buf = similar(baseline) .* false
          buf = similar(baseline); buf .= zero(T)

          method!(buf, lhs, rhs, 1, 0)
          @test buf ≈ baseline; @test !any(isnan, buf)
          method!(buf, lhs, rhs, -1, 0)
          method!(buf, lhs, rhs, 1, 0)
          @test buf ≈ baseline; @test !any(isnan, buf)


          # test @view interface and \beta \neq 0
          inds = collect(3:5:50)
          baseline[inds, :] .+= 2.5 * @view(lhs[inds, :]) * rhs;

          method!(@view(buf[inds, :]), @view(lhs[inds, :]), rhs, 2.5, 1)
          @test buf ≈ baseline rtol=sqrt(eps(eltype(baseline))); @test !any(isnan, buf)
          method!(@view(buf[inds, :]), @view(lhs[inds, :]), rhs, -2.5, 1)
          method!(@view(buf[inds, :]), @view(lhs[inds, :]), rhs, 2.5, 1)
          @test buf ≈ baseline rtol=sqrt(eps(eltype(baseline))); @test !any(isnan, buf)
        end
      end
    end
  end

  @testset "fastdensesparsemul_outer" begin
    @testset for T in [Float64, Float32]
      @testset for method! in [fastdensesparsemul_outer!, fastdensesparsemul_outer_threaded!]
        @testset for trial in 1:10
          lhs = rand(T, 50, 100);
          rhs = sprand(T, 100, 1_000, 0.1);
          k = rand(1:size(rhs, 1))

          baseline = lhs[:, k:k] * rhs[k:k, :];

          buf = zeros(T, size(baseline))

          method!(buf, @view(lhs[:, k]), rhs[k, :], 1, 0)
          @test buf ≈ baseline; @test !any(isnan, buf)
          method!(buf, @view(lhs[:, k]), rhs[k, :], -1, 0)
          method!(buf, @view(lhs[:, k]), rhs[k, :], 1, 0)
          @test buf ≈ baseline; @test !any(isnan, buf)

          baseline .+= 2.5 * lhs[:, (k+1):(k+1)] * rhs[(k+1):(k+1), :];

          method!(buf, lhs[:, k+1], rhs[k+1, :], 2.5, 1)
          @test buf ≈ baseline; @test !any(isnan, buf)
          method!(buf, lhs[:, k+1], rhs[k+1, :], -2.5, 1)
          method!(buf, lhs[:, k+1], rhs[k+1, :], 2.5, 1)
          @test buf ≈ baseline; @test !any(isnan, buf)
        end
      end
    end
  end
end
