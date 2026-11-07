# Windows descriptor publication

Fix `3d1866d5` lets a service or hook client finish reading its original
descriptor while the owner publishes a replacement. Later readers see the
new descriptor. All 120 focused checks pass, including 12 new cases, and
strict type checks pass for Windows and Linux. The new installed artifact
and full regression are pending.

Both descriptor consumers now open Windows files with read, write and
delete sharing. The writer first uses the existing replacement operation.
On a Windows sharing or access-denied failure, it tries
`SetFileInformationByHandle` with replacement and POSIX rename semantics.
Microsoft documents these semantics as retaining existing handles to the
old file while subsequent opens receive the replacement.
[Windows rename semantics](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information),
[Win32 rename structure](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_rename_info).

The temporary file receives its private ACL before publication. The change
does not remove the destination first, relax permissions, bypass a read-only
attribute or extend the existing 250 ms retry limit. Unsupported extended
rename operations return to that bounded legacy retry path. Readers that
do not share deletion can still cause publication to fail within the
existing deadline. Non-Windows reads and replacement retain their normal
Python file operations.

The held-reader tests cover both actual consumers and ordinary and Unicode
paths, including a character requiring two UTF-16 code units. They hold a
read handle open through publication, then verify old and new snapshots,
non-inheritance, closure and temporary-file cleanup. Other new checks cover
UTF-8 and missing files, decode failures, both reader ownership transfers,
three unsupported-operation codes and source-handle cleanup after a failed
native rename. Existing checks cover ordinary-reader retry limits, competing
launchers, private paths, service behavior, hooks and the WinHTTP boundary.

The original reproduction failed twice. Delete-sharing reads alone also
failed twice, with 11 other checks passing. After adding the rename fallback,
all 13 checks in that initial group passed. The expanded focused run passed
120 cases in 59.89 seconds. Black passed on all four changed files, and
strict mypy passed on all three changed modules for both target platforms.
The [receipt](descriptor-sharing-v1-receipt.json) records the source and
private archive, which preserves these attempts.

The [preceding full-suite failure](full-regression-preview-guidance-v1.md)
remains a failure on its original source. The
[native Codex v11 pass](native-codex-v11.md) also retains its original
artifact binding. These focused checks do not establish fresh native-host,
quality, latency or daily-use pilot results. No live configuration changed
and nothing was published.
