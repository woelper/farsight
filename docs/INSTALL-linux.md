# Install: Linux desktop input (M6)

`frontend-ibus` is a per-user IBus engine. It talks to `predictd` over its
usual socket and renders suggestions as preedit ghost text, so it works in
every IBus client (GTK, Qt, Electron, terminals) with no per-app code.

## 1. Build and install the binaries

```sh
cargo build --release -p predictd -p frontend-ibus
sudo install -m755 target/release/predictd /usr/local/bin/
sudo install -m755 target/release/frontend-ibus /usr/local/bin/
```

`predictd` must run as your user (systemd `--user` unit or `scripts/start.sh`).

## 2. Register the component

Copy the component file and refresh the registry cache, then restart IBus:

```sh
sudo cp crates/frontend-ibus/predict.xml /usr/share/ibus/component/
sudo ibus write-cache --system
ibus restart
```

Notes, both verified the hard way against ibus 1.5.29:

- The **system** component dir is the only one this daemon scans
  (`~/.local/share/ibus/component` is ignored — confirmed by syscall
  tracing). A user-level install path does not exist on this build.
- `predict.xml`'s `<exec>` must point at the installed engine binary; the
  daemon spawns exactly that on activation.
- The engine also self-registers (`RegisterComponent`) on every startup,
  which is what links the `predict` engine name to its factory. Activation
  needs **both**: the catalog entry (this step) and the running engine.

## 3. Select the engine

```sh
ibus engine predict
```

or pick *Predict* in `ibus-setup` → Input Method → Add. Type anywhere:
ghost continuations appear inline, `Tab` accepts, the lookup table offers
word candidates. Password and terminal fields stay silent — toolkits
declare them via `Properties.Set(ContentType)`, which the engine honors
(no queries, no learning, no display).

## 4. Verify

```sh
ibus engine            # -> predict
ibus read-config       # daemon healthy
```

End-to-end behavior (activation, ghost, commit, password silence,
slow-daemon safety) is covered by `crates/frontend-ibus/tests/`:

```sh
# engine-direct (needs a private daemon socket):
ibus-daemon --daemonize --panel=disable --address=unix:path=/tmp/ibus-test-sock
IBUS_TEST_ADDRESS=unix:path=/tmp/ibus-test-sock \
  cargo test -p frontend-ibus --test ibus_engine
# full client path, hermetic (spawns its own daemon, needs no env):
cargo test -p frontend-ibus --test ibus_mediated
```

GTK/Qt client verification is manual (no headless toolkit here): open
`gedit`, select *Predict*, type a few words, confirm ghost + Tab, then a
password field (`seahorse`, browser login) and confirm silence.

## Troubleshooting

- `Cannot find engine predict` after `ibus engine predict`: the catalog
  entry is missing — redo step 2 (check `ibus write-cache --system`
  ran after copying the XML).
- No suggestions, keys pass through: `predictd` unreachable
  (`PREDICTD_SOCKET`, or the default `~/.local/share/predict/predictd.sock`).
  The engine never blocks typing — it degrades to a pass-through.
- Stale ghost: `ibus restart` re-reads the registry; the engine resets on
  focus change.
