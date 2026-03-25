JULIA ?= julia

.PHONY: test test-serial benchmark format deps update help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

JOBS ?= 10

test: ## Run all tests in parallel (via ParallelTestRunner)
	$(JULIA) --project=test test/runtests.jl --jobs=$(JOBS)

test-serial: ## Run all tests serially (for debugging)
	$(JULIA) --project=test test/runtests.jl --jobs=1

benchmark: ## Run benchmarks
	$(JULIA) --project=benchmark benchmark/benchmarks.jl

format: ## Format all Julia files with Runic
	runic --inplace src/ test/ benchmark/

deps: ## Instantiate all environments
	$(JULIA) --project -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=test -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=benchmark -e 'using Pkg; Pkg.instantiate()'

update: ## Update all environments
	$(JULIA) --project -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=test -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=benchmark -e 'using Pkg; Pkg.update()'
