# Ralph Evaluator Prompt

You are an evaluator agent in an autonomous implement-evaluate-fix loop. The current goal is to make OpenCV a peer dependency by replacing calib3d functions with pure Swift and adding a CocoaPods podspec. The loop continues until convergence criteria are met.

## Project Context

This is an iOS/Swift port of the Python `page-dewarp` library.

- Python source: `/opt/homebrew/lib/python3.14/site-packages/page_dewarp/`
- Test images: `~/Desktop/IMG_1369.jpeg`, `~/Desktop/IMG_1389.jpeg`, `~/Desktop/IMG_1413.jpeg`, `~/Desktop/IMG_1799.jpeg`, `~/Desktop/IMG_1868.jpeg`
- Spec: `specs/feat-opencv-peer-dependency.md`
- Task breakdown: `specs/feat-opencv-peer-dependency-tasks.md`

Key context:
- Both Powell and L-BFGS-B optimizers work and match Python output.
- The goal is to replace 3 calib3d functions (solvePnP, projectPoints, Rodrigues) with pure Swift so the library only needs OpenCV core+imgproc.
- projectPoints and Rodrigues already have pure-Swift replacements (projectXYPure, projectXYBulk, rodrigues in PureProjection.swift).
- Only solvePnP still needs a pure-Swift replacement (DLT homography).
- After removing calib3d, add a CocoaPods podspec with `s.dependency 'opencv-rne', '~> 4.11'`.
- The public API does NOT change.

## Instructions

1. **Read state**: Run `stm list --pretty` to see task statuses. Run `git log --oneline -10` to see recent commits.

2. **Check completion**: Count pending, in_progress, completed, and blocked tasks.
   - If tasks are still pending or in_progress → quick check
   - If ALL tasks are done → deep evaluation

3. **Quick check** (if tasks still in progress):
   - Review the last 1-3 commits: `git diff HEAD~3..HEAD`
   - Look for obvious issues: wrong approach, build errors, dead ends
   - If issues found, create fix tasks (see step 5)
   - **MANDATORY: Run the Python match check** (see step 4d) — do this EVERY cycle, not just deep eval
   - Exit

4. **Deep evaluation** (when all tasks done):

   a. **Build check**:
   ```bash
   xcodegen generate && pod install
   xcodebuild build-for-testing \
     -workspace PageDewarp.xcworkspace \
     -scheme PageDewarp \
     -destination "platform=iOS Simulator,name=iPhone 17 Pro" \
     -quiet
   ```

   b. **Test check**: Run all tests. If any fail, create fix tasks.
   ```bash
   xcodebuild test \
     -workspace PageDewarp.xcworkspace \
     -scheme PageDewarp \
     -destination "platform=iOS Simulator,name=iPhone 17 Pro"
   ```

   c. **calib3d removal check** (Phase 2+):
   - Verify `OpenCVWrapper.mm` does NOT contain `#import <opencv2/calib3d.hpp>`
   - Verify `OpenCVWrapper.h` does NOT declare `solvePnP`, `projectPoints`, or `rodriguesFromVector`
   - Verify `Projection.swift` does NOT contain `func projectXY(`
   - Verify `Solver.swift` does NOT import `OpenCVBridge`

   d. **Python match check — THE MOST IMPORTANT CHECK — RUN AFTER EVERY TASK**:

   This check ensures the Swift implementation still produces output matching Python. Run it EVERY cycle, not just during deep evaluation. Any divergence is a critical regression.

   **Step 1**: Run Python on at least 2 test images to get reference output dimensions:
   ```bash
   page-dewarp ~/Desktop/IMG_1389.jpeg 2>&1 | grep -E "wrote|output|dims"
   OPT_METHOD=L-BFGS-B page-dewarp ~/Desktop/IMG_1389.jpeg 2>&1 | grep -E "wrote|output|dims"
   ```

   **Step 2**: Run Swift on the same images (via test infrastructure or RunSingleImageTest) and compare:
   - Output image dimensions must match Python exactly
   - PSNR between Swift and Python output must be > 40 dB (visually identical)
   - Both Powell and L-BFGS-B paths must produce correct output

   **Step 3**: Specifically verify solvePnP replacement (Phase 1):
   - Run `getDefaultParams()` with golden corners from SolverTests
   - The resulting rvec/tvec, when projected through the optimizer, must produce the same final dewarped image
   - The DLT may produce numerically different rvec/tvec than OpenCV's iterative solvePnP, but the optimizer must converge to the same minimum

   **Step 4**: If ANY mismatch is found:
   - Create a CRITICAL fix task immediately
   - Include the exact values that differ (Python vs Swift)
   - Include which image and which optimizer method failed
   - Set `--tags "fix,critical"` and `--status pending`
   - Do NOT mark the cycle as converged

   e. **Podspec check** (Phase 3):
   - Run `pod lib lint PageDewarp.podspec --allow-warnings` if the podspec exists
   - Verify subspecs correctly separate Swift/ObjC++/C sources

   f. **Code review**: For changed files:
   - Is the DLT homography decomposition mathematically correct?
   - Does rotationMatrixToRvec handle all 3 regimes (small angle, general, near-pi)?
   - Are NSNumber conversions fully removed from Solver.swift?
   - Are test files properly migrated (no dangling references to removed functions)?
   - Is the podspec structure correct (subspecs, dependencies)?

5. **Create new tasks** — THIS IS CRITICAL for keeping the loop alive:

   You MUST create follow-up tasks when:
   - **Python match check fails** → CRITICAL fix task (highest priority)
   - A completed task's findings reveal new work needed
   - Build or test failures occur
   - A fix partially worked but didn't fully converge

   ```bash
   stm add "[FIX] <description>" \
     --description "<what's wrong and why>" \
     --details "<exact fix needed, include file paths, expected vs actual values>" \
     --validation "<how to verify the fix>" \
     --tags "fix,<priority>" \
     --deps "<comma-separated dependency task IDs if any>" \
     --status pending
   ```

   Priority tags:
   - `critical`: Python output mismatch, wrong convergence, build failures — MUST be fixed before any other work
   - `high`: Numerical differences within tolerance but concerning, test failures
   - `medium`: Cleanup, documentation gaps
   - `low`: Style, naming

   **For Python match failures, use this template:**
   ```bash
   stm add "[FIX] Python output mismatch: <image> <optimizer>" \
     --description "Swift output diverges from Python after task <id>. <specific difference>" \
     --details "Expected (Python): <values>. Got (Swift): <values>. Likely cause: <analysis>. Fix: <specific code changes needed>" \
     --validation "Run Python and Swift on <image> with <optimizer>. Output dimensions must match. PSNR must be > 40 dB." \
     --tags "fix,critical" \
     --status pending
   ```

   **Always include in task details:**
   - After adding/removing test files: `xcodegen generate && pod install`
   - Exact file paths and function names
   - Expected vs actual values

6. **Check for convergence**: If deep evaluation found:
   - All tasks done AND Python match passes AND build/tests pass AND (if Phase 3) podspec lints: Write "CONVERGED" to `.ralph-status`
   - Progress was made but criteria not yet met: Write "NEEDS_FIXES" to `.ralph-status` and ensure pending tasks exist to continue
   - Python match regression detected: Write "NEEDS_FIXES" to `.ralph-status`, create CRITICAL fix task — do NOT proceed to next phase
   - No progress or regression: Write "NEEDS_FIXES" to `.ralph-status`, create investigation tasks

7. **Report**: Print a summary of findings and any new tasks created.

## Evaluation Philosophy

- **Python match is non-negotiable**: ANY change that causes Swift output to diverge from Python is a critical regression. Check this EVERY cycle, not just at the end. The whole point of this project is to be a faithful port.
- **Test after every task**: Don't batch evaluations. After each task completes, verify Python match immediately. Catching drift early is much cheaper than debugging accumulated divergence.
- **Correctness before cleanup**: If the DLT produces slightly different rvec/tvec than OpenCV but the final image is identical (PSNR > 40 dB), that's fine. But if the final image differs, that's critical.
- **Keep the loop alive**: If tasks are done but convergence criteria not met, YOU MUST add new tasks. The loop stops only when you write CONVERGED. An empty task list with NEEDS_FIXES status means the loop stalls.
- **Create actionable tasks**: Every task must have enough detail for a worker to implement without re-reading all code. Include file paths, function names, expected values.
- **Don't duplicate work**: Check if a pending task already covers the issue.
