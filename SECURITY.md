# Security Policy

## Supported versions

Security fixes are released for the latest minor version of the latest major version of
recsyslearn. Please upgrade before reporting.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately through GitHub:
[Report a vulnerability](https://github.com/giuliowaitforitdavide/recsyslearn/security/advisories/new).
If you cannot use GitHub, write to <giuliowaitforitdavide@duck.com>.

Include, as far as you can:

- the version of recsyslearn, Python and your operating system;
- a minimal input or script that triggers the problem;
- what happens (crash, memory error, wrong access) and what you expected.

The maintainer commits to replying as soon as possible. Once the problem is confirmed, a fix is
prepared privately, released, and then disclosed in a GitHub security advisory that
credits you unless you prefer otherwise.

## Scope

recsyslearn is a library that computes metrics on data passed by its caller; it performs
no network or file access. Reports that are in scope include memory-safety problems in
the native extension, crashes or hangs caused by crafted input, and compromised
dependencies or release artifacts. A wrong numerical result is a bug, not a
vulnerability: please use the bug report template for it.

The rules contributors follow to keep the code secure are in
[docs/development/security.md](docs/development/security.md).
