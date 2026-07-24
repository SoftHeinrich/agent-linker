# Incomplete E2E attempt

This run is not a five-project comparison and must not be reported as one.

It stopped during the S24 TeaStore controller call. Three controller evidence
quotes were exact document substrings, while a fourth shortened the source
sentence. The previous all-or-nothing grounding check raised because one quote
was invalid.

The redesign retains only exact grounded quotes and proceeds when at least one
exact quote supports the action. It still fails closed when none are grounded;
paraphrases never become tool evidence.
