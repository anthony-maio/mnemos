# Partial source review during the fixed v2 run

Status: three completed source inventories reviewed. The run was still active at review time. These observations are not a final score, a representative sample or an execution-policy amendment.

One inventory correctly preserved proposed status and nonapproval, with a supported whole-fact assessment matching the original source. Its temporal meaning remained unresolved. The relation/value decomposition was redundant but introduced no new assertion.

A second inventory retained a security-review prerequisite in its value text but returned an empty condition list. The source producer required governing conditions as separate quotations, while the query prompt treats no conditions as applicable. This is a structured condition-extraction failure even though the text itself still includes the prerequisite.

A third inventory preserved the reporter and lack of independent verification in two claims, but marked the whole-fact assessments and inventory unresolved. The original fact was itself an attributed, unverified report and matched the source verbatim. Independent proof that the reported event happened was not needed to support that qualified fact. The unresolved output therefore loses support availability rather than establishing a false positive.

The review is bound to the three original journal event digests and the pre-execution grading targets. Private review artifact: `.venv/relation-inventory-source-prefix-review-v2.json`, SHA-256 `af8aebfda348bbfbf3610f4adb447f508d03a122a475233a9c476487dc610d62`. No producer code, prompt, threshold, label or request schedule changed. Full-run availability, answer scores and source review remain pending.
