# Venue-Neutral Manuscript

This directory contains the anonymous 11pt article source used for Gate 7A.

Preferred compilation:

    latexmk -pdf main.tex

The project validation script also supports the locally cached Tectonic engine:

    bash ../../scripts/gate7a_compile_manuscript.sh

Compilation writes only to build/. The manuscript uses venue-neutral packages and does not include publisher or conference class files.
