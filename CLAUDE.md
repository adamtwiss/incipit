# CLAUDE.md — Incipit

Incipit is built with **no knowledge taken directly from other chess engines**. These rules are the project. Breaking one quietly does more damage than any bug. **If you are unsure whether something is allowed, stop and ask the human owner.**

## 1. What you may not consult
While working on Incipit, do not read or open:
- **Any other chess engine's source code, commit history, issues, or documentation**, whether on disk, on GitHub, or anywhere on the web. This applies to **every** engine regardless of license, **including other engines by Incipit's author (e.g. Coda, GoChess)**.
- **Pages/docs describing a specific engine's internals**, even on general sites such as the Chess Programming Wiki.
- **Another engine's networks, tuned parameters, or training data.**
- **Another project's agent memory, skills, or project directory** (e.g. Coda's), wherever they are on disk. Incipit agents use only Incipit's own project directory and memory.

Searching the web for a technique is fine, but don't open results that are an engine's code, docs or internals.

## 2. What you may consult
- **General technique descriptions** in the public literature: Chess Programming Wiki (or similar) articles about techniques (not about specific engines), academic papers, textbooks. Small code fragments in them may be read to understand an idea, but **never copied**. Close the source and implement from the idea in Incipit's own structures. If a fragment is attributed to a specific engine, skip it, and don't follow links from technique pages to engine pages.
- **Chess knowledge and rules:** piece values in the textbook sense (a rook is worth about five pawns), the fifty-move rule, standard chess/opening theory.
- **Approved tools only.** The approved list will live in a separate private repo for internal design notes, which doesn't exist yet. Until it does, nothing beyond what is already in the repo is approved: ask the human owner before adding any dependency or tool. Some libraries are themselves derived from engine code.

## 3. Ideas from the human owner
The human owner may suggest ideas in conversation, **including concepts known from other engines** ("should we consider dropping killer moves — many engines no longer use them"). That is ordinary engine development and is allowed.
- Suggestions are **ideas, never code**: implement them from the description. Don't go and look up how any engine implements the idea.
- If a suggestion comes with a specific constant, treat it like any other outside value (section 4).

**Not allowed: any form of mechanical harvesting.** Agents must not systematically survey other engines for ideas — no walking commit logs, changelogs, release notes or testing-framework histories to collect techniques, and no "what does engine X do that we don't" comparisons.

Incipit's ideas come from the human owner, first principles, the general literature, and Incipit's own measurements/experiments.

## 4. Constants and data
- **Never copy a tuned value from another engine**, or any dataset from anywhere, without explicit approval from the human owner.
- **Standard values are fine as starting points:** textbook chess values (e.g. piece values) and small, obvious defaults (e.g. a null-move verification search at depth − 2).
- For anything more specific, start from a principled or round value derived from Incipit's own quantities, note the derivation in a comment, and **tune it with SPSA**. Don't start from a detailed value you have seen in a paper or wiki example.
- All training data comes from Incipit's own self-play. The first generation is bootstrapped from material plus mobility, with weights Incipit tunes itself.

## 5. If another engine's source code enters your context
Stop. Don't write any code informed by it. Tell the human owner exactly what you saw and where, and ask for your session to be reset before continuing Incipit work.

This is about another engine's source code. Seeing an engine's name (in a list, a match result, or a technique page) is not a trigger.

## 6. Other engines as opponents
- **Allowed:** playing matches against other engines, comparing results and black-box metrics (Elo, NPS, EBF, depth, UCI output, hardware counters on a running binary), and discussing algorithms in general terms.
- **Not allowed:** disassembly, decompilation, or reading symbols to infer another engine's implementation.
- **Naming:** other engines are not named in Incipit's code, comments, commit messages, branch names, or public docs without approval of the human owner. Notes in the private design-notes repo (once it exists) may name opponents in test results.

## 7. The language model itself

The models doing this work were trained on public material that almost certainly includes chess engine source code. These rules govern what agents consult while working; they cannot undo what a model learned in training. Don't try to recall another engine's implementation. Build from the technique's description and Incipit's own measurements.
