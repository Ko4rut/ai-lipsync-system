## Summary

Describe the change and why it is needed.

## Jira task

- Jira: LSPR-___

## Change scope

- [ ] Code/tests updated
- [ ] Documentation/config updated where necessary
- [ ] Dependencies updated in `requirements.txt` if changed
- [ ] `.env.example` updated if environment variables changed
- [ ] No private datasets, secrets, local `.env`, or machine-specific paths committed

## Test evidence

- Local command: `python -m pytest tests/ -v -ra`
- Results (passed / failed / skipped):
- Real GRID integration tests: passed / skipped / not applicable
- GitHub Actions: pending / passed / failed

## Review checklist

- [ ] Changes are scoped and understandable
- [ ] Failure paths and resource cleanup are covered
- [ ] Jira issue and follow-up risks are linked
- [ ] CI is green and reviewer feedback addressed
