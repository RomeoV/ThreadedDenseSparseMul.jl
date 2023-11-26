∈ₛ(a,b) = occursin(a,b)
function check_module_used(mod_string::String, prof_data::Vector)
  cache = Dict{Symbol, String}();
  sframes = Profile.getdict(prof_data) |> values |> Iterators.flatten
  used_paths = [Profile.short_path(frame.file, cache) for frame in sframes]
  any(path->mod_string ∈ₛ path, used_paths)
end
