# Extraction eligibility clarification development

Seventy-two real Qwen2.5:14b proposal observations compare the prior history
clarification with an added eligibility rule. Twelve authored cases run
three repetitions per arm, all with the same existing control fact. Seven
cases were previously observed; five new positive cases mix real preferences
or requirements with questions, fiction, hypotheses, or prior suggestions.
Inputs, criteria, both prompts, model digest, and settings were frozen first.
No production prompt or live setting changed.

The added rule prevents proposals for the pure fiction and unaccepted
assistant-suggestion controls in all three repetitions. It still produces
proposals for the hypothetical in all three, and introduces a false negative:
all three fiction-project responses reject the actual novel project and its
Markdown manuscript requirement as if the project itself were fictional.
The history-only arm retains that real requirement. Both arms abstain on the
question and retain proposals for the other positive cases.

The candidate correctly retains the explicit Helix preference despite an
appended question, the current staging port without adopting the hypothetical
alternative, and the user-adopted Fossil decision. These bounded observations
do not compensate for the lost writing-project requirement. Its signed-export
proposal also omits the explicit not-yet-built status, although it does not
claim the implementation exists. Inspect the preserved texts and criteria;
nonempty proposal counts in summary.json are not semantic accuracy scores.

All 72 responses are runtime-valid. The candidate is not adopted. This result
supports investigating model/inference capability and contract sensitivity
rather than assuming a longer prohibition repairs extraction. No service
facts were committed in this comparison, and the preceding service failures
and incomplete acceptance matrix remain unchanged.

The evidence index binds raw prompts, schemas, responses, provider metadata,
inputs, preparation/source hashes, helpers, and terminal status. This README
is outside the index. This is development evidence, not unseen evaluation or
independent human annotation.
