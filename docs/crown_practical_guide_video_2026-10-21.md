# AfterQuery "A practical guide for Crown experts" (intro video, received 2026-10-21)

Slide text and narration transcribed from the 6:45 video (12 slides, burned-in subtitles). Formatting only.

## Slides
1. Making your first strong task: Choose (a computation you understand and can verify), Build (a clear task, a
   correct answer, and meaningful tests), Improve (use evidence to decide what to revise). "You do not need a perfect
   first draft. You need a way to check it."
2. Decide these before writing the task: The result (what exactly should the solver compute?), The challenge (what
   could a capable non-specialist get wrong?), The evidence (how will you independently check the answer?). "Try
   writing one sentence for each before opening the skeleton."
3. Use the walkthrough as a working aid: Guide (current requirements), Walkthrough (how the pieces connect), Skeleton
   (download the comments with the files). "Keep the structure. Replace the toy calculation with your own science."
4. Build around one clear contract: Specification, Reference, Tests. "Substeps should be meaningful scientific
   results, not extra bookkeeping."
5. Hard is not the same as long. Useful scientific challenge: choosing a valid branch, handling a change of regime,
   stable numerical computation, connecting coupled quantities. Not a substitute for difficulty: hidden conventions,
   arbitrary precision demands, repeated trivial operations, unnecessary runtime. "Choose challenges that naturally
   belong to your scientific problem."
6. When a task is too easy, inspect the solution before making changes: Observe (what method did the model actually
   use?), Investigate (does it avoid a scientific decision your task should require?), Revise (add a justified regime
   or dependency, then revalidate). "A local model run is evidence, not a guarantee of platform difficulty."
7. Check what a wrong answer could get away with: Required outputs (would a partially correct answer pass?),
   Important inputs (could the solver ignore a parameter?), Allowed limits (what happens near a relevant boundary?).
   "A mutant should represent a plausible scientific mistake, not a crash."
8. Accept scientifically equivalent answers: Diagnose (separate equivalent representations from wrong physics),
   Correct (compare the relevant invariant or define a necessary convention), Recheck (keep genuine errors
   detectable). "Do not simply loosen every tolerance until everything passes."
9. Treat feedback as a diagnosis: Structure or setup (check paths, imports, file connections), Specification or
   correctness (clarify the rule or revisit the science), Difficulty (inspect how the model solved the task),
   Infrastructure (share the error; do not change the science to mask it). "Identify the failure first. Then make a
   targeted change."
10. Keep revisions small and explainable: Read, Change (smallest coherent fix), Verify. "Keep a short revision note:
    issue -> change -> check."
11. Ask for help in a way we can act on. What to send: task link or name, exact feedback, what you tried, the question
    you need answered. Example: "I added the analytic limit and updated the test. Is the remaining concern about the
    allowed inputs or the implementation?" "Flag conflicting feedback and suspected infrastructure problems early."
12. Your next working session: choose one result (define it and independently verify a small case), use the
    walkthrough (matching specification, reference and tests), review before submitting (alignment, meaningful
    difficulty, negative controls). "Aim for a task you can explain, defend, and improve."

## Narration (key sentences)
- If the challenge is only substituting numbers into supplied equations, the idea may need more depth.
- Start with a small, representative calculation before building the entire submission.
- You can use an AI assistant to help, but you must check its science, citations, code, and validation claims.
- A specialized topic can still be easy for a strong model. Difficulty should come from scientific decisions or
  implementation choices, not simply from length: selecting a physically valid branch, handling a regime change,
  preserving numerical stability, or combining coupled quantities. Only use these where the science calls for them.
- The requested result should be precise even when finding the method is challenging.
- When a model solves your task easily, inspect its solution before making changes. Look for a shortcut that is valid
  for every allowed input. A broader, scientifically justified regime or a meaningful dependency might require deeper
  reasoning. But if the model was simply correct, do not punish its method.
- Use the platform suggestions as possible directions, not guaranteed fixes. If you test locally, follow the current
  evaluation guidance and treat your results as indicative, not a promise of passing calibration.
- Ask whether a partially correct implementation could pass, whether an important input could be ignored, and
  whether allowed boundary cases behave as promised. A mutant failing because of a missing import tells you little.
- Good tests accept scientifically equivalent answers (e.g. eigenvector signs): compare an invariant; do not silently
  grade an accidental choice made by your reference.
- Repeatedly submitting the same task without understanding the feedback usually creates more work for reviewers.
  A revision request does not automatically mean the idea is bad.
- If several revisions are not resolving the same problem, stop and ask for a targeted clarification. Getting help at
  that point is more useful than guessing again. Send the task link or name, the exact feedback, what you tried, and a
  specific question.
- Local checks and the skeleton help you prepare, but do not guarantee approval or difficulty.
