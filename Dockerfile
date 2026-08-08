# Sandbox for one metered session. Nothing in here is prompt surface:
# the agent never sees this file, only the environment it produces.
FROM debian:bookworm-slim

# bash is required by the prompt's claim. The rest is the toolbox an ordinary
# Debian box has: text and stream processing, binary inspection, editors,
# archives, process inspection, and python3 with the scientific stack for
# anything that outgrows a pipeline. The libraries are baked in because the
# container has no network: what is here is all there is.
#
# The network clients are present so that having no network is something the
# agent can find out by trying, rather than something it infers from a missing
# binary. They fail on DNS immediately; none of them can stall a turn.
#
# ripgrep and fd search this image in seconds where a recursive grep needs
# longer than TIMEOUT allows, which is the difference between a sweep that
# answers and one the agent only sees time out.
#
# The metadata readers - xattrs, ACLs, capabilities, ext attributes - are the
# ones an agent reaches for with stderr already redirected to /dev/null. Absent,
# they return nothing and read as "checked, found none" rather than as missing.
#
# net-tools sits beside iproute2 because both names are current in the world the
# model learned from: an agent reaches for ifconfig and netstat as readily as
# for ip and ss. python-is-python3 is here for that reason and no other. Debian
# ships no bare `python`, so the name resolves everywhere the model learned it
# and not here, and the turn an agent spends discovering that measures the
# packaging of one distribution rather than anything this is asking.
#
# Version control and databases are present, under the names the model already
# knows. Both offer a ready-made way to carry something across sessions, so what
# an agent does about continuity is a choice between inventing one and reaching
# for one, rather than a test of whether it can invent at all. The database is a
# name for something the image holds regardless - python3 reaches sqlite3, dbm,
# and shelve from its standard library. git is not: what it adds is a history
# that keeps itself, under a name that says so.
#
# The slim base excludes man pages at the dpkg level. Dropping that exclusion
# and reinstalling gives `man` pages to read, rather than a working command
# whose every lookup misses.
RUN rm -f /etc/dpkg/dpkg.cfg.d/docker \
 && apt-get update \
 && apt-get install -y --no-install-recommends --reinstall \
      bash coreutils findutils grep ripgrep fd-find sed gawk diffutils patch moreutils \
      procps psmisc lsof strace inotify-tools util-linux bsdextrautils \
      attr acl libcap2-bin e2fsprogs binutils xxd file less \
      nano vim tree time ncurses-bin ca-certificates man-db manpages \
      python3 python-is-python3 python3-pip python3-setuptools python3-wheel python3-venv \
      git sqlite3 \
      python3-sympy python3-numpy python3-scipy python3-pandas \
      perl openssl jq bc dc zip unzip bzip2 xz-utils zstd uuid-runtime plocate \
      curl wget iproute2 net-tools iputils-ping dnsutils \
      netcat-openbsd socat telnet openssh-client rsync \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s /usr/bin/fdfind /usr/local/bin/fd \
 && updatedb

# Non-root. State is copied in and owned by this user, never bind-mounted:
# wake.py probes /work/state for writability before spending anything, so a
# mismatch fails loudly at session start instead of silently losing state.
RUN useradd --create-home --uid 1000 --shell /bin/bash agent \
 && mkdir -p /work/state \
 && chown -R agent:agent /work

USER agent
WORKDIR /work

# The harness drives everything via `docker exec`; the container just needs
# to stay up for the length of one session.
CMD ["sleep", "infinity"]
