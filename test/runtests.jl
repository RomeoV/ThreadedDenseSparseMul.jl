using ThreadedDenseSparseMul
using Test
import SparseArrays: mul!, sprand
import Profile

∈ₛ(a,b) = occursin(a,b)
# We use a hack to find out if Polyester is actually used, namely
# profiling the call and checking the used modules for Polyester.
# This only works if the problem size is large enough (and then is still a statistical property).
# I wasn't able to find a clear way, but probably there is one.
function check_module_used(mod_string::String, prof_data::Vector)
  cache = Dict{Symbol, String}();
  sframes = Profile.getdict(prof_data) |> values |> Iterators.flatten
  used_paths = [Profile.short_path(frame.file, cache) for frame in sframes]
  any(path->mod_string ∈ₛ path, used_paths)
end

@testset "Compare with equivalent dense mul" begin
  @testset for trial in 1:10
    lhs = rand(500, 1000);
    rhs = sprand(1000, 10_000, 0.1);

    baseline = lhs * Matrix(rhs);
    @test lhs * rhs ≈ baseline

    buf = similar(baseline)

    @testset "Check actually using Polyester" begin
        # make sure I've overwritten the regular mul correctly and actually use Polyester...
        Profile.clear(); Profile.@profile (buf .= lhs * rhs)
        @test check_module_used("@Polyester", Profile.fetch())
    end

    buf .= lhs * rhs
    @test buf ≈ baseline

    buf .= 0.
    buf .+= lhs * rhs
    @test buf ≈ baseline

    buf .= 0.
    buf .+= 2.0.*lhs * rhs
    @test buf/2 ≈ baseline

    mul!(buf, lhs, rhs, 1, 0)
    @test buf ≈ baseline
  end
end
