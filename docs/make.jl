using Documenter
using ThreadedDenseSparseMul

makedocs(
    sitename = "ThreadedDenseSparseMul",
    format = Documenter.HTML(),
    clean = true,
    checkdocs = :exports,
    modules = [ThreadedDenseSparseMul],
    repo = Remotes.GitHub("RomeoV", "ThreadedDenseSparseMul.jl");
    pages = [
        "index.md",
        # "api_reference.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/RomeoV/ThreadedDenseSparseMul.jl.git";
)
