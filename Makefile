JULIA ?= julia

# Certification lives in a separate subpackage so Arblib stays out of core.
CERT := lib/HomotopyContinuationNextCertification

.PHONY: test test-serial test-cert benchmark ttfx format deps update help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

JOBS ?= 10

test: ## Run all tests in parallel (core via ParallelTestRunner) + certification subpackage
	$(JULIA) --project=test test/runtests.jl --jobs=$(JOBS)
	$(MAKE) test-cert

test-cert: ## Run the certification subpackage test suite (threaded)
	$(JULIA) --project=$(CERT)/test -t 4 $(CERT)/test/runtests.jl

test-serial: ## Run all tests serially (for debugging)
	$(JULIA) --project=test test/runtests.jl --jobs=1
	$(JULIA) --project=$(CERT)/test $(CERT)/test/runtests.jl

test-threaded: ## Run solve tests with multiple threads (exercises Threaded executor)
	$(JULIA) -t auto --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'

benchmark: ## Run benchmarks
	$(JULIA) --project=benchmark benchmark/runbenchmarks.jl

ttfx: ## Measure first-call latency per workload, one fresh session each
	$(JULIA) --project=benchmark benchmark/runttfx.jl $(WORKLOADS)

compare: ## Compare all categories against HomotopyContinuation v2
	$(JULIA) --project=benchmark benchmark/compare/runcompare.jl

format: ## Format all Julia files with Runic
	runic --inplace src/ test/ benchmark/ $(CERT)/src/ $(CERT)/test/

deps: ## Instantiate all environments
	$(JULIA) --project -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=test -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=$(CERT) -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
	$(JULIA) --project=$(CERT)/test -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path="."), Pkg.PackageSpec(path="$(CERT)")]); Pkg.instantiate()'
	$(JULIA) --project=benchmark -e 'using Pkg; Pkg.instantiate()'

update: ## Update all environments
	$(JULIA) --project -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=test -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=$(CERT) -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=$(CERT)/test -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=benchmark -e 'using Pkg; Pkg.update()'
