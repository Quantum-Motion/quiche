# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?= -W -j auto
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
H2_URL 		  = https://portal.nersc.gov/cfs/m888/dcamps/hamlib/chemistry/electronic/standard/H2.zip

.PHONY: help clean data Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

# Fetch the tutorial dataset once (prevent data races in parallel builds)
data:
	curl -fsSL -o tutorials/H2.zip "$(H2_URL)"
	unzip -o tutorials/H2.zip H2.hdf5 -d tutorials

# Catch-all target: route all unknown targets to Sphinx.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
