# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

Use this section to tell people how to report a vulnerability.

Tell them where to go, how often they can expect to get an update on a
reported vulnerability, what to expect if the vulnerability is accepted or
declined, etc.

## Known dependency advisories

### `sqlitedict` 2.1.0 (GHSA-g4r7-86gm-pgqc)

`sqlitedict` is a transitive dependency of `lm-eval`. There is no fixed
upstream release. Heretic explicitly disables lm-eval's request cache
(`cache_requests=False` in `BenchmarkScore`), so the vulnerable pickle-based
cache path is not used. The dependency remains in the lockfile and Dependabot
will continue to report this advisory until `lm-eval` drops it.
