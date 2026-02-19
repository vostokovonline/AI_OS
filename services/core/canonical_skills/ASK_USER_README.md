# Ask User Skill - Documentation

## Overview

Intelligent user clarification system that asks questions only when necessary, with smart throttling and timeout handling.

## Features

✅ **AI-Powered Necessity Detection** - Analyzes if question is truly needed
✅ **Throttling** - Prevents excessive questions (10 per goal, 5 per hour)
✅ **Timeout Handling** - Auto-recovery when user doesn't respond
✅ **Priority System** - Critical questions bypass throttling
✅ **Flexible Actions** - Configure what happens on timeout

## Quick Start

```python
from canonical_skills.ask_user import AskUserSkill, QuestionRequest

skill = AskUserSkill()

# Simple question
request = QuestionRequest(
    question="What format should the report be in?",
    context="Creating analysis report, need user preference",
    goal_id="goal-123"
)

result = await skill.execute(request, llm_call=your_llm_function)

if result.success:
    question_id = result.output["question_id"]
    print(f"Question asked: {question_id}")
    print(f"Status: {result.output['status']}")
```

## Timeout Handling

### What happens after timeout?

When user doesn't respond within timeout period (default: 1 hour), the system takes action based on `timeout_action`:

#### 1. `continue_with_default` (Recommended)
Uses default answer and continues execution:

```python
request = QuestionRequest(
    question="Report format?",
    context="Creating analysis report",
    goal_id="goal-123",
    timeout_action="continue_with_default",
    default_answer="markdown"  # ← Used if timeout
)
```

**Result**: After timeout, uses "markdown" as answer and continues.

#### 2. `wait_longer`
Extends timeout by 1 hour (max 3 extensions):

```python
request = QuestionRequest(
    question="Complex decision needed",
    context="Need user input on architecture",
    goal_id="goal-123",
    timeout_action="wait_longer",
    timeout_seconds=7200  # 2 hours
)
```

**Result**: Extends timeout up to 3 times before failing.

#### 3. `fail_goal`
Marks the goal as failed:

```python
request = QuestionRequest(
    question="Critical approval needed",
    context="Cannot proceed without user consent",
    goal_id="goal-123",
    timeout_action="fail_goal",
    priority="critical"
)
```

**Result**: Goal marked as failed with error message.

## API Endpoints

### Get Pending Questions
```bash
GET /questions/pending?goal_id=goal-123

# Response:
{
  "status": "ok",
  "count": 2,
  "questions": [
    {
      "question": "What format?",
      "artifact_id": "uuid-123",
      "priority": "normal",
      "timeout_at": "2026-01-18T20:00:00",
      "timeout_action": "continue_with_default"
    }
  ]
}
```

### Answer Question
```bash
POST /questions/{question_id}/answer

# Request body:
answer=markdown

# Response:
{
  "status": "ok",
  "message": "Answer recorded successfully",
  "question": "What format?",
  "answer": "markdown"
}
```

### Question Statistics
```bash
GET /questions/stats

# Response:
{
  "status": "ok",
  "pending_count": 3,
  "priority_breakdown": {
    "critical": 0,
    "high": 1,
    "normal": 2,
    "low": 0
  }
}
```

### Check Timeouts (Manual)
```bash
POST /questions/check-timeouts

# Manually trigger timeout check
# (Automatically runs every 5 minutes)
```

## Timeout Flow

```
User asked question
    ↓
Wait for response (default: 1 hour)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│                 │                 │                 │
default_answer   wait_longer      fail_goal         answered
    ↓                 ↓                 ↓               ↓
Use default     Extend timeout   Goal failed      Process answer
Continue        (+1 hour, max 3)                   Continue
```

## Best Practices

### 1. Always provide default for non-critical questions
```python
request = QuestionRequest(
    question="Report format?",
    context="...",
    goal_id="...",
    timeout_action="continue_with_default",
    default_answer="markdown"  # ✅ Good
)
```

### 2. Use `fail_goal` only for critical approvals
```python
request = QuestionRequest(
    question="Confirm database deletion?",
    context="Cannot proceed without approval",
    goal_id="...",
    priority="critical",
    timeout_action="fail_goal"  # ✅ Critical only
)
```

### 3. Use `wait_longer` for complex decisions
```python
request = QuestionRequest(
    question="Choose architecture pattern?",
    context="Need time to consider options",
    goal_id="...",
    timeout_action="wait_longer",
    timeout_seconds=7200  # 2 hours
)
```

### 4. Set appropriate priority
```python
# Critical - bypasses throttling
priority="critical"  # Emergency, security issues

# High - asked sooner
priority="high"  # Important decisions

# Normal - default behavior
priority="normal"  # Most questions

# Low - asked when system is idle
priority="low"  # Nice-to-have clarifications
```

## Monitoring

### Check timeout status
```python
# In logs:
⏰ Question timeout: uuid-123
   Action: continue_with_default
   ✓ Used default answer
```

### Question history
```bash
GET /questions/history/{goal_id}

# See all Q&A for a goal
```

## Examples

### Example 1: Non-critical with default
```python
# "What format?" -> If no answer, use markdown
request = QuestionRequest(
    question="What format should I use for the report?",
    context="Creating analysis report for code review",
    goal_id="goal-abc-123",
    timeout_action="continue_with_default",
    default_answer="markdown",
    timeout_seconds=1800  # 30 minutes
)
```

### Example 2: Critical approval
```python
# "Confirm deletion?" -> If no answer, fail goal
request = QuestionRequest(
    question="Please confirm: delete production database?",
    context="CRITICAL: Database deletion requested",
    goal_id="goal-xyz-789",
    priority="critical",
    timeout_action="fail_goal",
    timeout_seconds=3600  # 1 hour
)
```

### Example 3: Complex decision
```python
# "Choose pattern?" -> If no answer, wait longer
request = QuestionRequest(
    question="Which architecture pattern: MVC or Microservices?",
    context="Need to decide system architecture",
    goal_id="goal-def-456",
    timeout_action="wait_longer",
    timeout_seconds=7200,  # 2 hours initially
    options=["MVC", "Microservices", "Monolith"]
)
```

## Troubleshooting

### Q: Question timed out but goal didn't continue
**A**: Check if `default_answer` is provided for `continue_with_default` action.

### Q: Too many questions being asked
**A**: Check throttling stats: `GET /questions/stats`
- May need to increase `MIN_TIME_BETWEEN_QUESTIONS`
- Or improve AI necessity detection

### Q: Goal failed unexpectedly
**A**: Check if `timeout_action="fail_goal"` is used correctly.
- Should only be used for critical questions
- Consider using `continue_with_default` instead

## Configuration

### Throttling Limits
```python
MAX_QUESTIONS_PER_GOAL = 10
MAX_QUESTIONS_PER_HOUR = 5
MIN_TIME_BETWEEN_QUESTIONS = 300  # 5 minutes
```

### Timeout Settings
```python
DEFAULT_TIMEOUT_SECONDS = 3600  # 1 hour
MAX_TIMEOUT_EXTENSIONS = 3  # For wait_longer
```

## Future Enhancements

- [ ] WebSocket support for real-time answers
- [ ] Question templates for common queries
- [ ] Machine learning for smarter necessity detection
- [ ] Integration with notification systems (email, slack)
- [ ] Question analytics dashboard
