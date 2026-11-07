# Private pilot configuration rollback rehearsal

All seven unapplied host patches reproduced their exact proposed byte hashes
on protected disposable copies. Rollback restored two existing files
byte-for-byte and restored the original absence of five new files. Each change
also passed deliberate-drift checks before application and before rollback:
all fourteen attempts were refused and the changed copy was preserved.

The rehearsal read the current live files only to verify the plan's original
hashes and make disposable copies. All nine live target/reference paths were
rechecked unchanged. No host, provider or memory service started. No live patch
was applied. Four full-configuration copies were removed after independent
verification; the archive contains receipts, hashes, patches and frozen helper
sources, not full user configuration.

The first preflight failed before creating its output directory. A reference
file directly under the user home caused the original boundary check to reject
all temporary directories within that home. The corrected check continues to
exclude actual host/project configuration directories, live target overlap and
existing output directories. The original failed runner and log are preserved.
All eight focused checks, formatting and strict types pass.

The [receipt](pilot-config-rollback-v2-receipt.json) binds source `094a0bf6`,
the exact private revision-3 plan and 17 verified archive members. Archive
SHA-256: `38aefd882d8bc1f319b88d6ed1a3f1fbbbfa52ad260991097c5c25e83f65f677`.

This is a configuration rehearsal with intentional edits on disposable copies.
It does not exercise concurrent live edits, replace the required fresh backups
before activation, or restore a memory database. The separately recorded
backup/restore check retains its own artifact binding and forget-suppression
rules. Live activation, normal host trust, current-artifact recovery and quality
gates remain open. The pilot is inactive with zero recorded days.
