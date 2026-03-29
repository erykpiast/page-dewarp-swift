# Development Process

This iOS/Swift port was created almost entirely by autonomous [Claude Code](https://claude.ai/code) agents. This document describes the process and what we learned.

## The Ralph Loop

The port was built using "Ralph" -- a two-agent autonomous loop:

```
               +---> Worker Agent ---+
               |     (implements)    |
               |                     v
         stm task queue          git commit
               ^                     |
               |                     v
               +---- Evaluator <-----+
                     (reviews)
```

**Worker** ([`scripts/ralph-worker.md`](../scripts/ralph-worker.md)): Picks a pending task from [STM](https://github.com/nicobailon/task-master-stm) (a CLI task manager), implements it, runs tests, commits. One task per invocation.

**Evaluator** ([`scripts/ralph-evaluator.md`](../scripts/ralph-evaluator.md)): Reviews recent work, runs the build, reads code, compares against the Python reference. Creates fix tasks for issues found.

**Loop driver** ([`scripts/ralph-loop.sh`](../scripts/ralph-loop.sh)): A shell script that alternates between worker and evaluator invocations until convergence (all tasks done, no new issues). Also includes a status checker ([`scripts/ralph-status.sh`](../scripts/ralph-status.sh)).

### Worker prompt highlights

- Reads both the Swift file being modified and the corresponding Python source before making changes
- Runs `xcodebuild test` after every implementation
- Creates conventional commits (`feat(ios):`, `fix(ios):`)
- One task per invocation to keep changes focused

### Evaluator prompt highlights

- Runs build checks, test checks, and code review
- Compares Swift code against Python algorithm line-by-line
- Triages issues by severity: critical (build failures, wrong math) through low (style)
- Can create PLAN tasks for complex issues that invoke the spec workflow (`/spec:create` -> `/spec:validate` -> `/spec:decompose`)

## Implementation Phases

### Phase 1: Initial Port (Tasks 1-17)

The evaluator agent analyzed the Python codebase and created 17 tasks covering:

1. Project setup (XcodeGen, CocoaPods, OpenCV dependency)
2. OpenCV ObjC++ bridge (all cv2 calls wrapped)
3. Core algorithm modules (contours, spans, keypoints, projection, optimization)
4. Pipeline orchestration (DewarpPipeline.swift)
5. Remapping and thresholding
6. Unit tests for each module
7. Integration tests with golden file comparison

All 17 tasks were completed autonomously. The initial implementation built and passed unit tests but had not been compared against Python output.

### Phase 2: Evaluation & Fixes (Tasks 18-26)

The evaluator found 9 issues after the initial implementation:

- Build errors from API mismatches
- Test failures in edge cases
- Algorithm bugs discovered by code review

All fixed autonomously by the worker.

### Phase 3: Output Parity (Tasks 27-35)

Running both implementations on the same photos revealed significant output differences. A specification was created (`/spec:create`) defining a stage-by-stage comparison pipeline, then decomposed into 9 tasks:

1. **EXIF orientation fix** -- The critical bug. `cv2.imread` applies EXIF rotation automatically; Swift's `CGImage` does not. Fixed in `OpenCVWrapper.mm` by rendering through a `UIGraphicsImageContext` that respects `UIImage.imageOrientation`.

2. **Stage-by-stage validation** -- Compared intermediate values at each pipeline stage (dimensions, contours, keypoints, initial params, objective function, optimizer output, final image).

3. **Optimizer convergence** -- Powell's method (gradient-free) finds slightly different minima than Python's L-BFGS-B (gradient-based with JAX autodiff). Accepted as a known gap after verifying the objective function itself is correct.

### Phase 4: Additional Fixes (Tasks 36-39)

The evaluator's deeper analysis found 4 more issues:

- **norm2pix truncation**: Swift was using `floor()` where Python uses `int()` (truncation toward zero). Different for negative numbers.
- **Powell direction rotation**: Direction set update didn't match SciPy's heuristic.
- **Line search tolerance**: Brent's method tolerance was too loose.
- **ContourDetector thickness check**: Column thickness check was ordered differently from Python, causing different contour filtering.

## Results

### Before fixes (after initial port)

| Image | Issue |
|-------|-------|
| IMG_1358 | Rotated 90 degrees, text still curved |
| IMG_1359 | Severely distorted, mostly blank |

### After fixes

Tested on 20 recipe book photos:
- **All 20 succeeded** (no crashes, no failures)
- **Pixel match with Python**: 81-96% (average ~89%)
- **Visually correct**: All outputs are flat, upright, readable
- **Dimension match**: Within 2-10% of Python

The remaining ~11% pixel difference is primarily due to the optimizer finding slightly different solutions (Powell vs L-BFGS-B).

## Task Statistics

| Metric | Count |
|--------|-------|
| Total tasks created | 39 |
| Original implementation | 17 |
| Evaluator-found fixes | 9 |
| Parity comparison tasks | 9 |
| Late-stage fixes | 4 |
| Tasks completed autonomously | 39/39 (100%) |
| Human interventions during loop | 0 |

## Lessons Learned

1. **EXIF orientation is a landmine.** The single biggest bug was the EXIF mismatch between `cv2.imread` (auto-rotates) and `CGImage` (raw sensor pixels). This caused cascading failures in every downstream stage. Testing with a real photo (not a synthetic test image) caught it immediately.

2. **Stage-by-stage comparison is essential.** Comparing final output is insufficient for debugging -- the image looks wrong but you can't tell why. Dumping intermediate values at each pipeline stage (dims, contours, params, loss) pinpointed exactly where values diverged.

3. **The optimizer matters more than the objective.** The objective function matched Python's within 1e-8 for the same inputs. But Powell's method (gradient-free, ~600K max evals) and L-BFGS-B (with JAX autodiff gradients, ~300 evals) converge to different solutions. The math is identical; the search strategy isn't.

4. **Autonomous loops need evaluation diversity.** The initial worker/evaluator loop caught build errors and test failures quickly but missed output quality issues. Adding visual comparison and numerical validation to the evaluator's checklist caught the remaining bugs.

5. **ObjC++ bridges are tedious but correct.** Every OpenCV call requires marshalling data between Swift types, Objective-C types, and C++ types. This is verbose but mechanically straightforward and preserves numerical precision. Moving the mask computation entirely into ObjC++ (instead of calling back to Swift between steps) was a pragmatic simplification.

## Tools Used

- **Claude Code** (Opus model) -- autonomous agent for implementation and evaluation
- **STM** (Simple Task Master) -- CLI task manager for tracking work
- **XcodeGen** -- project file generation from YAML
- **CocoaPods** -- OpenCV dependency management
- **Python page-dewarp** -- reference implementation for golden data comparison
