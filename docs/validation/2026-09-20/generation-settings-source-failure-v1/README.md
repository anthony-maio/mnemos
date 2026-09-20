# Generation settings checkpoint: source verification failure

Source 05c71d2 recorded 2,004 passes, four skips, and one failure in the complete
suite. test_private_descriptor_replace_recovers_after_reader_closes exhausted
the private descriptor replacement path with Windows error 5. Subsequent
package checks were not run because the helper stops on failure.

A separate unchanged-source diagnostic exercised 25 actual reader-close and
replacement cycles. All 25 passed. This does not establish the original cause
or erase the failed full-suite result. The retry deadline and tests were not
weakened. A fresh complete run uses separate output and temporary directories.
Both the original JUnit/log and diagnostic observations are retained here.
