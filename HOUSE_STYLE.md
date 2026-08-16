# Manuscript House Style

This guide records the writing and revision style developed for this paper. Apply it to all
new manuscript text and all text added in response to reviewers. Do not rewrite established
text solely for style unless the author asks for a broader language pass.

## Core Principle

Use simple, plain, direct English. Technical ideas should remain technically exact, but the
reader should not have to decode the prose before understanding the science.

Prefer concrete verbs and nouns over abstract or pseudo-technical phrasing.

- Write “the inverse temperature affects retrieval and direct sampling in different ways,”
  not “the inverse temperature affects two related but distinct objects.”
- Write “we compared four sampling procedures,” not “we constructed a four-rung sampler
  ladder.”
- Write “the samples lay farther from the curve,” not “off-manifold error increased,” unless
  the formal term is needed and immediately defined.

## Build the Scientific Story

Every analysis should guide the reader through five questions:

1. Why did we perform the analysis?
2. What specific question or hypothesis did it test?
3. What did we compare or calculate?
4. What happened?
5. What can and cannot be concluded?

Do not introduce a metric without explaining why it answers the question. Do not report a
result before giving the reader enough context to interpret it. Add a short transition when
the text moves to a new biological or technical question.

## Sentences and Terminology

- Keep sentences direct and reasonably short.
- Do not begin a sentence with an acronym. For example, use “Stochastic attention
  nevertheless...” rather than “SA nevertheless....”
- Do not use em dashes. Use a comma, colon, parentheses, or a new sentence.
- Use technical terms only when they add precision. Define them at first use in language a
  biomedical reader can follow.
- Avoid inflated transitions and claims such as “critically,” “remarkably,” “intrinsic,”
  “machinery,” “harness,” or “substantive answer” when a direct statement will do.
- Avoid metaphors for technical workflows unless they make the method easier to understand.
- Define all unfamiliar metrics on first use, including what higher or lower values mean.
- Prefer “test,” “analysis,” or “comparison” over “attack” for membership inference unless
  discussing the formal literature term.
- Use “profiles” or “patients” consistently and do not call generated profiles independent
  clinical observations.

## Equations and Mathematical Transitions

- Introduce every displayed equation with a complete sentence that says what the equation
  gives, defines, or shows, and end the sentence with a colon.
- Do not lead into a displayed equation with a dangling phrase or comma. For example, write
  “Completing the square in each exponent gives the expression:” before the equation.
- When displayed equations occur in sequence, connect them with a short sentence that
  explains how the second follows from the first.

## Figures and Tables

- Cite every main-text figure and table in the Results at the point where its evidence is
  first reported or interpreted. A reference in the Methods, caption, or Discussion does not
  replace the Results citation.
- Place the figure or table reference in the sentence that states the corresponding finding.
  Do not use a detached “see Figure” or “see Table” sentence.
- Use `Fig.~\ref{fig:label}` for one figure, `Figs.~\ref{fig:first}--\ref{fig:last}` for
  multiple figures, and `Table~\ref{tab:label}` for a table. Use “Supplementary Fig.” and
  “Supplementary Table” for supplementary items.
- Identify the relevant panel inline when a claim depends on one panel, for example
  `(Fig.~\ref{fig:label}A)`.
- Build each Results passage in this order: explain the question, state what was compared or
  calculated, report the finding with its inline figure or table reference, and interpret what
  the finding does and does not show.
- Before considering the manuscript complete, inventory every `fig:`, `sfig:`, `tab:`, and
  `stab:` label and confirm that it is cited in the appropriate Results passage. Flag orphaned
  floats, Results claims without supporting references, and figures or tables cited only in
  Methods, captions, or Discussion.

## Project-Specific Technical Language

- Stochastic attention was not fit to the cohort. Patient profiles were stored as columns of
  the memory matrix used during sampling. Use “constructed the memory matrix,” “stored,” or
  “used during sampling.”
- Baseline models may be described as fit or estimated when that is what was done.
- Distinguish the Hopfield retrieval update from the corresponding sampling distribution.
  For the unit-normalized memories used here, multiplicity weights set the component
  probabilities and inverse temperature sets the spread around the selected profile.
- Generating 100 subgroup profiles does not create 100 independent patients or add clinical
  evidence beyond the source cohort.
- The source data were de-identified. The membership-inference test asked whether an
  already-known de-identified profile had been included in the memory matrix. It did not
  identify a person or reconstruct an unknown patient.
- Distinguish what was demonstrated from what was not tested. Use “was consistent with” when
  the analysis did not establish causation.
- Do not present future work as a promise. State the current limitation and, when useful, the
  type of method that could address it.

## Tense

- Results describing completed analyses and observed findings must be in the past tense.
- Methods describing what was done should normally be in the past tense.
- Mathematical definitions and general properties may use the present tense.
- Discussion statements about the study’s findings should remain appropriately bounded and
  should not shift into stronger present-tense claims than the results support.

## Paragraphs and Flow

- Aim for paragraphs similar in length to the Results section, usually about 100 to 200 words.
- Merge adjacent short paragraphs when they answer the same question.
- Split a long paragraph when it contains more than one scientific purpose.
- Each paragraph should have one central job and a clear opening sentence.
- Preserve necessary detail, but move implementation detail to Methods or the Supplement
  when it interrupts the main narrative.
- End the Discussion with its limitations paragraph. Place computational scaling and other
  methodological scope considerations before that final paragraph.

## Claims and Limitations

- State the strongest claim supported by the analysis, but no stronger.
- A non-significant test does not establish equivalence.
- Reproducing a small training cohort is not the same as recovering its population.
- Mechanistic agreement for one modeled system does not validate unmodeled biological
  features.
- A generated point lying outside a convex hull does not by itself show implausible
  extrapolation in a sparse, high-dimensional cohort.
- Synthetic generation without a formal privacy mechanism is not anonymization and does not
  provide differential privacy.

## Reviewer Revisions

- Preserve the reviewer color macros: `\rone{...}`, `\rtwo{...}`, and `\rboth{...}`.
- Change only reviewer-added text unless the author explicitly approves edits to legacy
  prose.
- Keep the response-to-reviewers document aligned with the manuscript. It should explain
  what changed using the same interpretation and terminology as the paper.
- A response should answer the reviewer directly, then summarize the evidence and identify
  the manuscript location.

## Repository Workflow

For every manuscript revision:

1. Make the same textual change in `paper/` and `arxiv/`.
2. Update `peer-review-feedback/response-to-reviewers.md` when the change responds to a
   reviewer.
3. Preserve unrelated author edits.
4. Compile the manuscript and supplement with `make all` from `paper/`.
5. Run `git diff --check`.
6. Check the LaTeX logs for undefined references, overfull text, and other warnings.
7. Inspect the rendered passage when a change could affect layout or paragraph flow.

## Final Read-Aloud Test

Before considering revised prose complete, ask:

- Would a biomedical reader understand why this analysis was performed?
- Is every technical term necessary and explained?
- Does each sentence say exactly what happened?
- Is any claim stronger than the evidence?
- Could the same idea be stated more simply without losing precision?

If the answer to the last question is yes, simplify it.
