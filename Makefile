JULIA ?= julia

# `| tee` returns tee's status, which would hide a failing suite; the test
# recipes below need bash's pipefail for the suite to remain a real gate.
SHELL := /bin/bash

CERT := lib/HomotopyContinuationCertification

TEST_LOG ?= test-run.log
CERT_LOG ?= test-cert.log
EXTENSIVE_LOG ?= test-extensive.log

.PHONY: test test-serial test-cert test-extensive jet benchmark ttfx format deps update help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

JOBS ?= 10

test: ## Run all tests in parallel + certification subpackage (full log: test-run.log)
	set -o pipefail; $(JULIA) --project=test test/runtests.jl --jobs=$(JOBS) 2>&1 | tee $(TEST_LOG)
	$(MAKE) test-cert

test-cert: ## Run the certification subpackage test suite (log: test-cert.log)
	set -o pipefail; $(JULIA) --project=$(CERT)/test $(CERT)/test/runtests.jl --jobs=$(JOBS) 2>&1 | tee $(CERT_LOG)

test-extensive: ## Run long reference solves outside the default suite (log: test-extensive.log)
	set -o pipefail; $(JULIA) --project=test/extensive -t auto test/extensive/runtests.jl 2>&1 | tee $(EXTENSIVE_LOG)

test-serial: ## Run core + certification suites serially for debugging
	set -o pipefail; $(JULIA) --project=test test/runtests.jl --jobs=1 2>&1 | tee $(TEST_LOG)
	set -o pipefail; $(JULIA) --project=$(CERT)/test $(CERT)/test/runtests.jl --jobs=1 2>&1 | tee $(CERT_LOG)

jet: ## Run JET package and workload analysis in the test environment
	$(JULIA) --project=test -e 'using Pkg; Pkg.develop(PackageSpec(path=".")); Pkg.instantiate()'
	$(JULIA) --project=test test/quality/jet.jl

benchmark: ## Run runtime benchmarks
	$(JULIA) --project=benchmark benchmark/runtime/runbenchmarks.jl

ttfx: ## Measure first-call latency per workload in fresh Julia sessions
	$(JULIA) --project=benchmark benchmark/ttfx/runttfx.jl $(WORKLOADS)

format: ## Format Julia sources, tests, benchmarks, and quality scripts with Runic
	runic --inplace src/ ext/ test/ benchmark/ workloads/ $(CERT)/src/ $(CERT)/test/

deps: ## Instantiate all development environments
	$(JULIA) --project -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
	$(JULIA) --project=test/quality/strict_mode -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
	$(JULIA) --project=$(CERT) -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
	$(JULIA) --project=$(CERT)/test -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path="."), Pkg.PackageSpec(path="$(CERT)")]); Pkg.instantiate()'
	$(JULIA) --project=test/extensive -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path="."), Pkg.PackageSpec(path="$(CERT)")]); Pkg.instantiate()'
	$(JULIA) --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

update: ## Update all development environments
	$(JULIA) --project -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=test -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=test/quality/strict_mode -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=$(CERT) -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=$(CERT)/test -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=test/extensive -e 'using Pkg; Pkg.update()'
	$(JULIA) --project=benchmark -e 'using Pkg; Pkg.update()'
