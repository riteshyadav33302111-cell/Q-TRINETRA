# CI

`github-workflow-ci.yml` is the GitHub Actions workflow (No-ML gate + pytest + demo smoke).
The automation token used to build this repo lacks the `workflows` permission, so copy it manually:

```bash
mkdir -p .github/workflows && cp ci/github-workflow-ci.yml .github/workflows/ci.yml
git add .github && git commit -m "ci: enable GitHub Actions" && git push
```
