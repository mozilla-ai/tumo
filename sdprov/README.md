# sdprov — SD card fleet provisioning for the Pi 5 workshop

Provisions Raspberry Pi OS **Trixie** SD cards from macOS by editing only the
FAT32 boot partition (cloud-init `user-data` / `network-config` / `meta-data`),
then collects each board's WiFi MAC over the USB ethernet gadget and verifies
WiFi once the MACs are registered.

Requires: macOS, `uv` (or any Python ≥ 3.11), an SD reader, a USB-C data cable.

## Setup

```bash
cd sdprov
cp config.example.toml config.toml   # git-ignored
$EDITOR config.toml                  # WiFi creds, user, count, ssh key
```

Setting `ssh_authorized_key` is strongly recommended — without it every
`collect`/`verify` ssh connection prompts for the password.

## Per-card provisioning

Insert a card (macOS mounts the boot partition at `/Volumes/bootfs`) and:

```bash
uv run sdprov.py provision            # allocates the next free piNN
uv run sdprov.py provision --hostname probe   # special card, no fleet index
```

What it writes on the boot partition:

- `user-data` — hostname, workshop user + password (both `users:` for fresh
  images and `chpasswd:` for golden-image clones), ssh password auth +
  optional authorized key, timezone/keymap, and first-boot commands: WiFi
  regulatory country, `modprobe g_ether`, `rpi-usb-gadget on -f`, pin the
  WiFi MAC to the hardware address.
- `network-config` — netplan for the lab WiFi (DHCP, `optional: true`).
- `meta-data` — bumps `instance-id`, which makes cloud-init re-run first-boot
  setup (and regenerate ssh host keys) even on an already-booted clone.
- `ssh` — empty file, force-enables sshd on every boot.
- `config.txt` — ensures `dtoverlay=dwc2,dr_mode=peripheral` so the USB
  gadget works from the very first boot (same line `rpi-usb-gadget` manages).
- `sdprov-state.json` — marker so re-inserting a card re-provisions it under
  the same name instead of burning a new index (override with `--force`).

The card is ejected when done. Boot the Pi connected to the Mac's USB-C port
(one cable: power + data). With no Internet Sharing on the Mac, the Pi serves
DHCP on the link (10.12.194.1/28) and is reachable as `piNN.local`.

## MAC sweep with one probe card

The WiFi MAC belongs to the **board**, not the card, so one card can sweep
all 20 Pis:

```bash
uv run sdprov.py provision --hostname probe   # once, on the probe card
uv run sdprov.py collect --loop --host probe.local
```

The loop waits for each Pi to appear on USB, records serial + MACs into
`inventory.csv`, tells you what label to stick on the board, and moves on.
Then print the registration list:

```bash
uv run sdprov.py maclist
```

## Verify (at the lab, after MAC registration)

```bash
uv run sdprov.py verify            # all provisioned Pis: hostname, WiFi, internet
uv run sdprov.py verify --camera   # also require a detected CSI camera
```

## Golden image checklist

Before imaging the master card (see repo discussion):

1. `sudo rpi-usb-gadget on` once, so clones have the NM usb0 profiles baked.
2. `sudo nmcli connection delete "<your-home-ssid>"` — don't ship your WiFi
   password on 20 cards.
3. Same username in `config.toml` as the user on the golden image.
4. Optional: `sudo apt clean`, `rm -rf ~/.cache/uv`, clear shell history.
5. Do **not** wipe cloud-init state or ssh host keys manually — provision's
   `instance-id` bump re-runs per-card setup, and its `ssh_deletekeys`
   override regenerates host keys on each provisioned clone (RPi OS disables
   cloud-init key management by default, and its own regeneration only fires
   on an image's first boot ever — same for disk expansion, which provision
   also handles itself).
6. Shrink the image (PiShrink) so it flashes fast and fits every card.

After flashing a clone, run `provision` on it as usual.

## Creating the golden image file

1. Finalize the card (checklist above: `nmcli connection show`, delete every
   WiFi profile, `apt clean`, clear caches/history) and `sudo poweroff`.
2. Read the card into an image on the Mac (~10 min for 32 GB on USB-3):

   ```bash
   diskutil list                          # find the card, e.g. /dev/disk4 — CHECK THE SIZE
   sudo diskutil unmountDisk /dev/disk4
   sudo dd if=/dev/rdisk4 of=~/golden.img bs=4m status=progress
   diskutil eject /dev/disk4
   ```

3. Shrink + compress with PiShrink (Linux-only; use Docker Desktop):

   ```bash
   docker run --rm --privileged -v ~/:/work debian:stable-slim bash -c '
     apt-get update -qq && apt-get install -y -qq wget parted e2fsprogs xz-utils >/dev/null &&
     wget -qO /usr/local/bin/pishrink https://raw.githubusercontent.com/Drewsif/PiShrink/master/pishrink.sh &&
     chmod +x /usr/local/bin/pishrink &&
     pishrink -s -Z /work/golden.img'
   ```

   `-s` is required: it skips PiShrink's legacy auto-expand hack (pre-Trixie
   mechanism; cloud-init growpart re-expands the filesystem instead, triggered
   by provision's instance-id bump). `-Z` produces `golden.img.xz`, which
   rpi-imager consumes directly. No Docker? Run the same on a Pi against the
   .img on external storage.

4. Flash clones with Raspberry Pi Imager: Choose OS → **Use custom** →
   `golden.img.xz`. **Decline Imager's OS customization** — it cannot
   re-trigger cloud-init on an already-booted image (cached instance-id) and
   would silently half-apply. Run `uv run sdprov.py provision` on each card
   after flashing instead.
5. Validate the first clone before mass-flashing: boot it and check it comes
   up under its **new** hostname (re-personalization works), `df -h /` shows
   the filesystem filling the card, and there is no ssh host-key reuse
   warning versus another clone (key regeneration works).

Note on disk expansion: Trixie's own `rpi-resize.service` only fires on an
image's very first boot ever (`ConditionFirstBoot=yes`, i.e. machine-id not
yet initialized), so golden-image clones would never expand on their own.
provision therefore injects a `growpart`+`resize2fs` first-boot command —
expansion happens on every provisioned card automatically.

## Notes

- The boot partition is world-readable FAT: the WiFi PSK and (per cloud-init
  convention) the user password appear in plaintext there. Fine for a
  workshop network + throwaway credentials; don't reuse real passwords.
- `inventory.csv` accumulates board serials and MACs — that's the record you
  send for registration and the state `verify` checks against.
- Bookworm cards are rejected on purpose: pre-Trixie images have no cloud-init
  (their `custom.toml` only works on never-booted cards, and WiFi would need
  ext4 edits). Re-flash with Trixie instead.
