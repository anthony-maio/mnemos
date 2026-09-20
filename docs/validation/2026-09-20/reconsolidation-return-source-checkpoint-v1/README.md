# Reconsolidation return source/package checkpoint

Frozen product d937892 passes 2,013 source tests with four skips, formatting,
Python 3.10 compatibility type checking, JavaScript syntax, wheel/sdist build,
and twine metadata checks. All six checks have exit code zero. The clean
checkout, external test storage, runner hash, JUnit outcomes and package hashes
are recorded in verification.json and the accompanying logs.

This adds the regression proving that direct reconsolidation returns the
committed metadata after absorbing a captured duplicate. The installed-package
check is separate and is not claimed by this archive. The earlier 26c9ddb
installed result does not validate this newer artifact. Full-module quality,
broader host recovery and the daily-use pilot remain open.
