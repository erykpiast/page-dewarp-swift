# Ralph Worker Prompt

You are a worker agent in an autonomous implement-evaluate-fix loop. Your job is to pick one pending task from STM, implement it fully, test it, commit, and mark it done.

## Project Context

This is an iOS/Swift port of the Python `page-dewarp` library. The current goal is to make Swift's L-BFGS-B optimizer converge to the same result as Python's L-BFGS-B.

Key findings from prior work:
- Powell optimizer works correctly — Swift matches Python Powell output
- Swift has both optimizers: `DewarpPipeline.process(image:method:)` with `.powell` or `.lbfgsb`
- The analytical gradient in AnalyticalGradient.swift is confirmed correct (< 1e-4 vs finite differences)
- BUT Swift L-BFGS-B converges to a different local minimum than Python L-BFGS-B:
  - Python L-BFGS-B: rvec[0]≈0.053, pageDims 1.43x2.20
  - Swift L-BFGS-B: rvec[0]≈0.184, pageDims 1.60x3.06
- The L-BFGS-B C library (Sources/CLBFGSB/) is a different implementation from scipy's Fortran code
- Objective function is verified correct (matches Python value at initial params)

Python source is at: `/opt/homebrew/lib/python3.14/site-packages/page_dewarp/`
Key Python files: `projection.py`, `solve.py`, `image.py`, `optimise/_scipy.py`, `optimise/_base.py`, `keypoints.py`

## Debug Output

**IMPORTANT**: Save ALL debug output, traces, comparison reports, and intermediate files to `~/Desktop/lbfgsb-debug/`. Create the directory if it doesn't exist (`mkdir -p ~/Desktop/lbfgsb-debug/`). The user monitors this directory for progress.

## Available Skills

You have access to these skills which can help with complex tasks:
- `/brainstorm` — Use when you need to explore multiple hypotheses or approaches before picking one
- `/spec:create` — Use when you need to plan a complex implementation before coding

Don't hesitate to use these if the task involves non-obvious decisions.

## Instructions

1. **Read state**: Run `stm list --pretty` to see all tasks and their statuses.

2. **Pick the next task**: Find a `pending` task whose dependencies (listed in `--deps`) are ALL `completed` or `done`. If multiple qualify, prefer:
   - `critical` priority before `high` before `medium`
   - Smaller tasks to build momentum
   - If no pending tasks with satisfied deps exist, report "BLOCKED" and exit
   - **IMPORTANT**: Skip any task that is already `in_progress` — another worker is on it.

3. **Mark it in-progress immediately**: Run `stm update <id> --status in_progress` BEFORE doing any work. This prevents other parallel workers from picking the same task. If `stm update` fails (task already in_progress), go back to step 2 and pick a different task.

4. **Read the task details**: Run `stm show <id>` to get full implementation details.

5. **Read relevant source**: Before modifying any module, ALWAYS read both:
   - The Swift file being modified
   - The corresponding Python source in `/opt/homebrew/lib/python3.14/site-packages/page_dewarp/`
   Do not work from memory — read the actual code.

6. **Implement the task**: Follow the details exactly. Key rules:
   - Use Double (float64) for all numerical computation
   - Use the OpenCV bridge pattern for OpenCV calls (ObjC++ wrapper)
   - Reference the Python source with inline comments: `// Ported from projection.py:36-47`
   - Save all debug/trace/comparison output to `~/Desktop/lbfgsb-debug/`
   - When comparing Swift vs Python behavior, use the comparison script or test infrastructure

7. **Verify**: Build and test:
   ```bash
   # Generate project and install pods if needed
   xcodegen generate && pod install

   # Build
   xcodebuild build-for-testing \
     -workspace PageDewarp.xcworkspace \
     -scheme PageDewarp \
     -destination "platform=iOS Simulator,name=iPhone 17 Pro" \
     -quiet

   # Run all tests
   xcodebuild test \
     -workspace PageDewarp.xcworkspace \
     -scheme PageDewarp \
     -destination "platform=iOS Simulator,name=iPhone 17 Pro"
   ```

8. **Running the Swift pipeline on disk images** (for comparison/validation):
   The library is iOS-only. To process images from disk:
   - Copy input to `/tmp/` (simulator shares host's `/tmp` on Apple Silicon)
   - Create a test file that loads from `/tmp`, runs DewarpPipeline.process(), writes output to `/tmp`
   - Run with `xcodebuild test -only-testing:...`
   - After adding/removing test files, always run `xcodegen generate && pod install`
   - See `docs/running-on-disk-images.md` for full details

   To compare with Python:
   - Default: `page-dewarp <image-path>`
   - With L-BFGS-B: `OPT_METHOD=L-BFGS-B page-dewarp <image-path>`

   Test images are at: `~/Desktop/IMG_1369.jpeg`, `~/Desktop/IMG_1389.jpeg`, `~/Desktop/IMG_1413.jpeg`, `~/Desktop/IMG_1799.jpeg`, `~/Desktop/IMG_1868.jpeg`

9. **Commit**: Stage and commit with a conventional commit message:
    ```
    fix(optimizer): <description>
    ```
    Before committing, run `git pull --rebase` to incorporate any parallel workers' commits.

10. **Mark completed**: Run `stm update <id> --status completed`

11. **Report**: Print a summary of what you did, what worked, and any concerns for the evaluator.

## Important

- Do NOT attempt multiple tasks in one invocation. One task per run.
- If you hit a blocker, mark the task as `blocked` with a note and exit.
- If a task needs something not in STM, create a new task with `stm add`.
- Keep commits small and focused on the single task.
- Do NOT modify files outside the task scope.
- Do NOT break the Powell optimizer path — it must continue to work.
