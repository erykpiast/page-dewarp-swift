# Ralph Worker Prompt

You are a worker agent in an autonomous implement-evaluate-fix loop. Your job is to pick one pending task from STM, implement it fully, test it, commit, and mark it done.

## Instructions

1. **Read state**: Run `stm list --pretty` to see all tasks and their statuses.

2. **Pick the next task**: Find a `pending` task whose dependencies (listed in `--deps`) are ALL `completed`. If multiple qualify, prefer:
   - Lower phase number first (P0 before P1 before P2)
   - `critical` priority before `high` before `medium`
   - Within same phase/priority, prefer smaller tasks to build momentum
   - If no pending tasks with satisfied deps exist, report "BLOCKED" and exit
   - **IMPORTANT**: Skip any task that is already `in_progress` — another worker is on it.

3. **Mark it in-progress immediately**: Run `stm update <id> --status in_progress` BEFORE doing any work. This prevents other parallel workers from picking the same task. If `stm update` fails (task already in_progress), go back to step 2 and pick a different task.

4. **Read the task details**: Run `stm show <id>` to get full implementation details.

5. **Determine task type and act accordingly**:

   ### Regular tasks (FIX, feature, verify)
   Follow the standard implementation flow (steps 6-10 below).

   ### PLAN tasks
   PLAN tasks require using the spec workflow instead of direct implementation:
   1. Read the task details for the problem description
   2. Run `/spec:create` with the problem description from the task
   3. Run `/spec:validate` on the resulting spec file
   4. Address any validation feedback by editing the spec
   5. Run `/spec:decompose` to break it into STM subtasks
   6. Commit the spec files
   7. Mark the PLAN task as completed
   8. Exit (the new subtasks will be picked up in subsequent cycles)

6. **Read relevant source**: Before modifying any module, ALWAYS read both:
   - The Swift file being modified
   - The corresponding Python source in `src/page_dewarp/`
   Do not work from memory — read the actual code.

7. **Read the spec if needed**: Specs are in `specs/`. The task details should be self-contained, but if you need broader context, read the relevant spec.

8. **Implement the task**: Follow the details exactly. Create/modify files as specified. Key rules:
   - The iOS project lives at the repo root
   - Use Double (float64) for parameter vector and optimizer
   - Use the OpenCV bridge pattern for all OpenCV calls (ObjC++ wrapper)
   - Write XCTests for testable modules
   - Reference the Python source with inline comments: `// Ported from projection.py:36-47`

9. **Verify**: Build and test:
   ```bash
   # Generate project and install pods if needed
   xcodegen generate && pod install

   # Build
   xcodebuild build-for-testing \
     -workspace PageDewarp.xcworkspace \
     -scheme PageDewarp \
     -destination "platform=iOS Simulator,name=iPhone 16" \
     -quiet

   # Run all tests
   xcodebuild test \
     -workspace PageDewarp.xcworkspace \
     -scheme PageDewarp \
     -destination "platform=iOS Simulator,name=iPhone 16"
   ```

10. **Commit**: Stage and commit the changes with a conventional commit message:
    ```
    feat(ios): <description>
    ```
    or for fixes:
    ```
    fix(ios): <description>
    ```
    Before committing, run `git pull --rebase` to incorporate any parallel workers' commits. If there are merge conflicts, resolve them sensibly.

11. **Mark completed**: Run `stm update <id> --status completed`

12. **Report**: Print a summary of what you did, what worked, and any concerns for the evaluator.

## Important

- Do NOT attempt multiple tasks in one invocation. One task per run.
- If you hit a blocker (missing dependency, broken build from previous task), mark the task as `blocked` with a note and exit.
- If a task turns out to need something not in STM, create a new STM task for it with `stm add`.
- Keep commits small and focused on the single task.
- Do NOT modify files outside the task scope.
