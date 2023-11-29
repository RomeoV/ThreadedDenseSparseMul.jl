using ThreadedDenseSparseMul
using Test
import SparseArrays: mul!, sprand
import Profile


@testset "Compare with equivalent dense mul" begin
  @testset "fastdensesparse" begin
    @testset for trial in 1:10
      lhs = rand(500, 1000);
      rhs = sprand(1000, 10_000, 0.1);

      baseline = lhs * Matrix(rhs);

      buf = similar(baseline)

      fastdensesparse!(buf, lhs, rhs, 1, 0)
      @test buf ≈ baseline
      fastdensesparse!(buf, lhs, rhs, -1, 0)
      fastdensesparse_threaded!(buf, lhs, rhs, 1, 0)
      @test buf ≈ baseline


      # test @view interface and \beta \neq 0
      inds = collect(3:5:100)
      baseline[inds, :] .+= 2.5 * @view(lhs[inds, :]) * Matrix(rhs);

      fastdensesparse!(@view(buf[inds, :]), @view(lhs[inds, :]), rhs, 2.5, 1)
      @test buf ≈ baseline rtol=sqrt(eps(eltype(baseline)))
      fastdensesparse!(@view(buf[inds, :]), @view(lhs[inds, :]), rhs, -2.5, 1)
      fastdensesparse_threaded!(@view(buf[inds, :]), @view(lhs[inds, :]), rhs, 2.5, 1)
      @test buf ≈ baseline rtol=sqrt(eps(eltype(baseline)))
    end
  end

  @testset "fastdensesparse_outer" begin
    @testset for trial in 1:10
      lhs = rand(500, 1000);
      rhs = sprand(1000, 10_000, 0.1);
      k = rand(1:size(rhs, 1))

      baseline = lhs[:, k:k] * Matrix(rhs)[k:k, :];

      buf = similar(baseline)

      fastdensesparse_outer!(buf, @view(lhs[:, k]), rhs[k, :], 1, 0)
      @test buf ≈ baseline
      fastdensesparse_outer!(buf, @view(lhs[:, k]), rhs[k, :], -1, 0)
      fastdensesparse_outer_threaded!(buf, @view(lhs[:, k]), rhs[k, :], 1, 0)
      @test buf ≈ baseline

      baseline .+= 2.5 * lhs[:, (k+1):(k+1)] * Matrix(rhs)[(k+1):(k+1), :];

      fastdensesparse_outer!(buf, lhs[:, k+1], rhs[k+1, :], 2.5, 1)
      @test buf ≈ baseline
      fastdensesparse_outer!(buf, lhs[:, k+1], rhs[k+1, :], -2.5, 1)
      fastdensesparse_outer_threaded!(buf, lhs[:, k+1], rhs[k+1, :], 2.5, 1)
      @test buf ≈ baseline
    end
  end
end
