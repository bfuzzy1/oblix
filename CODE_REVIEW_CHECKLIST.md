# Code Review Checklist

> "Peer review = craftsmanship review" - Japanese-Level Code Quality Rule-Book

## 🎯 Monozukuri (Craftsmanship) Review

### Function Quality
- [ ] **Function size ≤ 50 lines** - Is the function focused and readable?
- [ ] **Complexity ≤ 10** - Is the logic straightforward?
- [ ] **Clear, full words** - No clever abbreviations like `usrLst` → `processedUsers`
- [ ] **Single responsibility** - Does the function do one thing well?

### Code Clarity
- [ ] **Self-documenting names** - Can a new team member understand the intent?
- [ ] **Business rules in comments** - Are complex business logic explained?
- [ ] **No magic numbers** - Are constants named and explained?
- [ ] **Consistent formatting** - Does the code follow project standards?

## 🔄 Kaizen (Continuous Improvement)

### Technical Debt
- [ ] **No "for later" comments** - Is technical debt addressed now?
- [ ] **No dead code** - Are unused functions/variables removed?
- [ ] **No speculative complexity** - Is the code solving today's need?
- [ ] **Sustainable pace** - Can this be maintained for 20 years?

### Learning & Growth
- [ ] **Knowledge sharing** - Are complex decisions documented?
- [ ] **Cross-functional understanding** - Can non-experts follow the logic?
- [ ] **Future-proof** - Will this code be understandable in 2035?

## 🎨 Wabi-sabi (Graceful Imperfection)

### Simplicity
- [ ] **Simplest solution** - Is this the most straightforward approach?
- [ ] **Easy to evolve** - Can future changes be made easily?
- [ ] **No premature optimization** - Is performance optimization justified?
- [ ] **Minimal configuration** - Are toggles only added for real use-cases?

### Maintainability
- [ ] **Boring, proven tech** - Are we using stable, well-understood patterns?
- [ ] **Vertical growth** - Are we extending existing modules before creating new abstractions?
- [ ] **Clear dependencies** - Are imports and dependencies obvious?

## 🚫 Eliminate the Seven Wastes

### Check for Waste
- [ ] **Partially done work** - Is the feature complete and tested?
- [ ] **Extra features** - Are we building only what's needed?
- [ ] **Re-learning** - Is knowledge captured and shared?
- [ ] **Handoffs** - Are we minimizing context switching?
- [ ] **Delays** - Are feedback loops under 24 hours?
- [ ] **Task switching** - Is the work focused and uninterrupted?
- [ ] **Defects** - Are tests covering edge cases?

## 🎯 Quality Gates

### Zero-Defect Policy
- [ ] **Tests pass** - Do all tests run successfully?
- [ ] **Edge cases covered** - Are weird edge cases that burned us before tested?
- [ ] **No warnings** - Does the code pass linting without warnings?
- [ ] **Coverage maintained** - Are new features properly tested?

### Code Review Questions
- [ ] **Clarity over cleverness** - Is the code clear rather than clever?
- [ ] **Maintainable** - Can someone else maintain this code?
- [ ] **Documented** - Are complex business rules explained?
- [ ] **Consistent** - Does it follow established patterns?

## 📋 Review Process

### Before Review
- [ ] **Self-review** - Have I reviewed my own code first?
- [ ] **Tests written** - Are tests covering the happy path and edge cases?
- [ ] **Documentation updated** - Are README/CHANGELOG updated?
- [ ] **Small, reviewable slices** - Is the diff ≤ 200 lines?

### During Review
- [ ] **Focus on craftsmanship** - Am I looking for clarity and maintainability?
- [ ] **Ask questions** - Am I seeking understanding, not criticism?
- [ ] **Suggest improvements** - Am I offering constructive feedback?
- [ ] **Celebrate good work** - Am I acknowledging quality craftsmanship?

### After Review
- [ ] **Update CHANGELOG** - Are changes documented immediately?
- [ ] **Share learnings** - Are insights shared with the team?
- [ ] **Plan next kaizen** - What's the next small improvement?

---

## 🎯 Daily Kaizen Reminder

> "The fastest way to go fast is to **never slow down**."

**Today's kaizen target:** Pick ONE small improvement:
- [ ] Better variable name
- [ ] Add missing comment
- [ ] Extract tiny helper function
- [ ] Improve error message
- [ ] Update documentation

**Remember:** 1% better every day compounds into excellence over time.