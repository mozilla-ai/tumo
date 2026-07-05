#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""sdprov — provision a fleet of Raspberry Pi 5 SD cards from macOS.

Targets Raspberry Pi OS Trixie (Debian 13) images, which read cloud-init
config from the FAT32 boot partition — so everything here works on the
natively-mounted /Volumes/bootfs, no ext4 access needed.

Subcommands:
  provision  customize the inserted SD card (hostname, user, WiFi, ssh,
             USB ethernet gadget) and record it in the inventory
  collect    ssh into a booted Pi (over the USB gadget link) and record
             its board serial + MAC addresses in the inventory
  verify     ssh into provisioned Pis and check hostname/WiFi/internet
  maclist    print the WiFi MAC list for network registration

Workflow: see README.md next to this file.
"""

import argparse
import csv
import datetime
import json
import plistlib
import re
import shlex
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.toml"
DEFAULT_INVENTORY = HERE / "inventory.csv"
STATE_FILE = "sdprov-state.json"  # marker we leave on the boot partition
OVERLAY_LINE = "dtoverlay=dwc2,dr_mode=peripheral"  # exact line rpi-usb-gadget uses

INVENTORY_FIELDS = [
    "hostname", "index", "provisioned_at", "collected_at", "verified_at",
    "board_serial", "model", "wlan0_mac", "eth0_mac", "throttled",
    "wifi_ip", "notes",
]

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=8",
]


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def die(msg: str) -> "NoReturn":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- config


def load_config(path: Path) -> dict:
    if not path.exists():
        die(f"config not found: {path}\n  cp {HERE / 'config.example.toml'} {path} and edit it")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for section, keys in {
        "fleet": ["hostname_prefix", "count", "username", "password"],
        "wifi": ["ssid", "password", "country"],
    }.items():
        for key in keys:
            if not cfg.get(section, {}).get(key) and cfg.get(section, {}).get(key) != 0:
                die(f"config: [{section}] {key} is required")
    country = cfg["wifi"]["country"]
    if not re.fullmatch(r"[A-Z]{2}", country):
        die("config: [wifi] country must be a 2-letter code like AM, IT, US")
    iso_tab = Path("/usr/share/zoneinfo/iso3166.tab")
    if iso_tab.exists():
        valid = {line.split("\t")[0] for line in iso_tab.read_text().splitlines()
                 if line and not line.startswith("#")}
        if country not in valid:
            hint = " (the United Kingdom is GB, not UK)" if country == "UK" else ""
            die(f"config: [wifi] country '{country}' is not a valid ISO 3166 code{hint} — "
                "an invalid country leaves WiFi rfkill-blocked on the Pi")
    tz = cfg.get("locale", {}).get("timezone")
    if tz and not (Path("/usr/share/zoneinfo") / tz).is_file():
        die(f"config: [locale] timezone '{tz}' does not exist (e.g. Europe/Paris, Asia/Yerevan)")
    if not re.fullmatch(r"[a-z][a-z0-9-]*", cfg["fleet"]["hostname_prefix"]):
        die("config: [fleet] hostname_prefix must be lowercase letters/digits/hyphens")
    if not (8 <= len(cfg["wifi"]["password"]) <= 63):
        die("config: [wifi] password must be 8-63 characters (WPA passphrase)")
    return cfg


def hostname_for(cfg: dict, index: int) -> str:
    width = max(2, len(str(cfg["fleet"]["count"])))
    return f"{cfg['fleet']['hostname_prefix']}{index:0{width}d}"


# ------------------------------------------------------------- inventory


def read_inventory(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("hostname"):
                    rows[row["hostname"]] = {k: row.get(k, "") or "" for k in INVENTORY_FIELDS}
    return rows


def write_inventory(path: Path, rows: dict[str, dict]) -> None:
    def sort_key(r: dict):
        return (0, int(r["index"])) if r.get("index") else (1, r["hostname"])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        for row in sorted(rows.values(), key=sort_key):
            writer.writerow(row)


def upsert(rows: dict[str, dict], hostname: str, **fields) -> dict:
    row = rows.setdefault(hostname, {k: "" for k in INVENTORY_FIELDS})
    row["hostname"] = hostname
    for k, v in fields.items():
        if v is not None:
            row[k] = str(v)
    return row


def allocate_index(cfg: dict, rows: dict[str, dict]) -> int:
    for i in range(1, cfg["fleet"]["count"] + 1):
        if hostname_for(cfg, i) not in rows:
            return i
    die(f"all {cfg['fleet']['count']} hostnames are already in the inventory "
        "(pass --hostname to go beyond, or edit inventory.csv)")


# ---------------------------------------------------------- boot volume


def find_boot_volume(explicit: str | None) -> Path:
    if explicit:
        vol = Path(explicit)
        if not (vol / "config.txt").exists():
            die(f"{vol} does not look like a Raspberry Pi boot partition (no config.txt)")
        return vol
    candidates = [
        v for v in Path("/Volumes").iterdir()
        if (v / "config.txt").exists() and (v / "cmdline.txt").exists()
    ]
    if not candidates:
        die("no Raspberry Pi boot partition found under /Volumes — is the SD card inserted?")
    if len(candidates) > 1:
        die(f"multiple boot partitions found: {', '.join(map(str, candidates))} — pass --volume")
    return candidates[0]


def check_trixie_cloudinit(vol: Path) -> None:
    issue = vol / "issue.txt"
    if issue.exists():
        print(f"  image: {issue.read_text().splitlines()[0].strip()}")
    if not (vol / "meta-data").exists() and not (vol / "user-data").exists():
        die(
            f"{vol} has no cloud-init files (user-data/meta-data).\n"
            "  This looks like a pre-Trixie (Bookworm or older) image, which sdprov does\n"
            "  not support — re-flash the card with Raspberry Pi OS Trixie (2025-11-24+)."
        )


def eject_volume(vol: Path) -> None:
    info = plistlib.loads(
        subprocess.run(
            ["diskutil", "info", "-plist", str(vol)],
            check=True, capture_output=True,
        ).stdout
    )
    disk = info.get("ParentWholeDisk")
    if not disk:
        die(f"could not determine the disk for {vol}; eject manually with Finder")
    subprocess.run(["diskutil", "eject", disk], check=True)


# --------------------------------------------------------- file contents


def q(s: str) -> str:
    """JSON-encode a string; a JSON string is a valid YAML double-quoted scalar."""
    return json.dumps(s)


def render_user_data(cfg: dict, hostname: str) -> str:
    fleet, wifi, locale = cfg["fleet"], cfg["wifi"], cfg.get("locale", {})
    key = (fleet.get("ssh_authorized_key") or "").strip()

    # First-boot commands. rpi-usb-gadget only writes modules-load.d (next
    # boot), so modprobe g_ether first to bring usb0 up in this very boot.
    # The nmcli tweak pins the WiFi MAC to the hardware one, so the address
    # we registered with the lab is guaranteed to be the one that associates.
    cmds = [
        f"raspi-config nonint do_wifi_country {wifi['country']}",
        "rfkill unblock wifi",
        "modprobe g_ether",
        "rpi-usb-gadget on -f",
        f"nmcli connection modify {shlex.quote('netplan-wlan0-' + wifi['ssid'])} "
        "802-11-wireless.cloned-mac-address permanent",
        # Diagnostic report on the boot partition — readable from any laptop
        # when the Pi is unreachable. Runs last, after the commands above.
        "{ date; echo '--- cloud-init:'; cloud-init status --long; "
        "echo '--- udc (empty = dwc2 overlay not active):'; ls /sys/class/udc; "
        "echo '--- modules:'; lsmod | grep -E 'dwc2|ether|libcomposite'; "
        "echo '--- links:'; ip -o link; "
        "echo '--- addrs:'; ip -o -4 addr; "
        "echo '--- nm devices:'; nmcli -t device; "
        "echo '--- nm connections:'; nmcli -t connection; "
        "echo '--- wifi:'; nmcli -t -f ACTIVE,SSID,SIGNAL dev wifi; "
        "echo '--- rfkill:'; rfkill list; "
        "echo '--- NetworkManager journal:'; journalctl -b -u NetworkManager --no-pager | tail -30; "
        "echo '--- cloud-init errors:'; grep -iE 'error|warn|traceback' /var/log/cloud-init.log | tail -30; } "
        "> /boot/firmware/sdprov-debug.log 2>&1",
    ]

    lines = [
        "#cloud-config",
        "# generated by sdprov — regenerated on every provision run",
        f"hostname: {q(hostname)}",
        "manage_etc_hosts: true",
    ]
    if locale.get("timezone"):
        lines.append(f"timezone: {q(locale['timezone'])}")
    if locale.get("keymap"):
        lines += ["keyboard:", f"  layout: {q(locale['keymap'])}"]
    lines += [
        "users:",
        f"- name: {q(fleet['username'])}",
        "  groups: users,adm,dialout,audio,netdev,video,plugdev,cdrom,games,input,gpio,spi,i2c,render,sudo",
        "  shell: /bin/bash",
        "  lock_passwd: false",
        f"  plain_text_passwd: {q(fleet['password'])}",
        '  sudo: ["ALL=(ALL) NOPASSWD:ALL"]',
    ]
    if key:
        lines += ["  ssh_authorized_keys:", f"  - {q(key)}"]
    # users: only takes effect at user creation; chpasswd also covers the
    # golden-image case where the user already exists on the cloned rootfs.
    lines += [
        "chpasswd:",
        "  expire: false",
        "  users:",
        f"  - name: {q(fleet['username'])}",
        f"    password: {q(fleet['password'])}",
        "    type: text",
        "ssh_pwauth: true",
        "runcmd:",
    ]
    for cmd in cmds:
        lines.append(f"- [sh, -c, {q(cmd + ' || true')}]")
    return "\n".join(lines) + "\n"


def render_network_config(cfg: dict) -> str:
    wifi = cfg["wifi"]
    lines = [
        "# generated by sdprov",
        "network:",
        "  version: 2",
        "  renderer: NetworkManager",
        "  wifis:",
        "    wlan0:",
        "      dhcp4: true",
        "      optional: true",
        "      access-points:",
        f"        {q(wifi['ssid'])}:",
        f"          password: {q(wifi['password'])}",
    ]
    if wifi.get("hidden"):
        lines.append("          hidden: true")
    return "\n".join(lines) + "\n"


def bump_instance_id(vol: Path, hostname: str) -> str:
    """Rewrite instance-id in meta-data, preserving everything else.

    A new instance-id makes cloud-init treat the card as a fresh instance and
    re-run per-instance setup (hostname, users, ssh host-key regeneration) —
    this is what makes provisioning work on golden-image clones that have
    already booted.
    """
    iid = f"sdprov-{hostname}-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    meta = vol / "meta-data"
    kept = []
    if meta.exists():
        kept = [
            line for line in meta.read_text().splitlines()
            if not re.match(r"^\s*instance[-_]id\s*:", line)
        ]
    if not any(line.strip() and not line.lstrip().startswith("#") for line in kept):
        kept = ["dsmode: local"] + [l for l in kept if l.strip()]
    kept.append(f"instance-id: {iid}")
    meta.write_text("\n".join(kept) + "\n")
    return iid


def ensure_config_txt_overlay(vol: Path) -> bool:
    """Ensure the dwc2 peripheral overlay is active from the first boot.

    Uses the exact line rpi-usb-gadget writes; the script dedupes it, so the
    two never conflict.
    """
    cfg_txt = vol / "config.txt"
    text = cfg_txt.read_text()
    if re.search(rf"^{re.escape(OVERLAY_LINE)}$", text, re.M):
        return False
    if not text.endswith("\n"):
        text += "\n"
    cfg_txt.write_text(text + OVERLAY_LINE + "\n")
    return True


# -------------------------------------------------------------- ssh side


def ssh_capture(cfg: dict, host: str, remote_cmd: str) -> dict[str, str] | None:
    """Run a key=value-emitting command over ssh; None if unreachable/failed."""
    user = cfg["fleet"]["username"]
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, f"{user}@{host}", remote_cmd],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        print(f"  ssh failed: {err[-1] if err else f'exit {proc.returncode}'}")
        return None
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def wait_for_ssh(host: str, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, 22), timeout=3):
                return True
        except OSError:
            time.sleep(3)
    return False


def describe_throttled(raw: str) -> str:
    try:
        bits = int(raw, 16)
    except (ValueError, TypeError):
        return ""
    notes = []
    if bits & 0x1:
        notes.append("UNDER-VOLTAGE NOW")
    if bits & 0x10000:
        notes.append("under-voltage occurred")
    if bits & 0x2 or bits & 0x20000:
        notes.append("throttling")
    return ", ".join(notes)


COLLECT_CMD = r"""
echo "board_serial=$(tr -d '\0' < /sys/firmware/devicetree/base/serial-number 2>/dev/null)"
echo "model=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
echo "wlan0_mac=$(cat /sys/class/net/wlan0/address 2>/dev/null)"
echo "eth0_mac=$(cat /sys/class/net/eth0/address 2>/dev/null)"
echo "throttled=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)"
echo "hostname=$(hostname)"
"""

VERIFY_CMD = r"""
echo "hostname=$(hostname)"
echo "wifi_ip=$(ip -4 -o addr show wlan0 2>/dev/null | awk '{print $4}' | head -n1)"
echo "ssid=$(nmcli -t -f ACTIVE,SSID dev wifi 2>/dev/null | awk -F: '$1=="yes"{print $2; exit}')"
echo "internet=$(curl -m 8 -sI http://deb.debian.org/ >/dev/null 2>&1 && echo ok || echo fail)"
echo "camera=$(rpicam-hello --list-cameras 2>/dev/null | grep -qE '^[0-9]+ :' && echo ok || echo none)"
echo "throttled=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)"
"""


# ------------------------------------------------------------ provision


def cmd_provision(args, cfg: dict) -> None:
    vol = find_boot_volume(args.volume)
    print(f"provisioning card at {vol}")
    check_trixie_cloudinit(vol)

    rows = read_inventory(args.inventory)

    state_path = vol / STATE_FILE
    previous = None
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            pass

    if args.hostname:
        hostname, index = args.hostname, None
        m = re.fullmatch(rf"{re.escape(cfg['fleet']['hostname_prefix'])}0*(\d+)", args.hostname)
        if m:
            index = int(m.group(1))
    elif previous and previous.get("hostname") and not args.force:
        hostname, index = previous["hostname"], previous.get("index")
        print(f"  card was already provisioned as {hostname} — re-provisioning "
              "with the same name (use --force to allocate a new one)")
    else:
        index = allocate_index(cfg, rows)
        hostname = hostname_for(cfg, index)

    (vol / "user-data").write_text(render_user_data(cfg, hostname))
    (vol / "network-config").write_text(render_network_config(cfg))
    iid = bump_instance_id(vol, hostname)
    added_overlay = ensure_config_txt_overlay(vol)
    (vol / "ssh").write_text("")  # sshswitch enables ssh on every boot; belt & suspenders

    state_path.write_text(json.dumps(
        {"hostname": hostname, "index": index, "instance_id": iid, "provisioned_at": now()},
        indent=2,
    ) + "\n")

    upsert(rows, hostname, index=index or "", provisioned_at=now())
    write_inventory(args.inventory, rows)

    print(f"  wrote user-data        (hostname={hostname}, user={cfg['fleet']['username']})")
    print(f"  wrote network-config   (wifi ssid={cfg['wifi']['ssid']}, country={cfg['wifi']['country']})")
    print(f"  wrote meta-data        (instance-id={iid})")
    print(f"  wrote ssh              (force-enable ssh)")
    print(f"  config.txt overlay     ({'added' if added_overlay else 'already present'}: {OVERLAY_LINE})")
    print(f"  inventory updated      ({args.inventory})")

    if args.eject:
        eject_volume(vol)
        print(f"  ejected — boot the Pi with this card, connected to the Mac via USB-C")
    else:
        print(f"  not ejected (--no-eject); eject before removing the card")
    print(f"\nnext: sdprov.py collect --host {hostname}.local"
          + (f" --index {index}" if index else ""))


# -------------------------------------------------------------- collect


def collect_one(cfg: dict, rows: dict, host: str, hostname_key: str,
                index: int | None, inventory: Path, wait: int) -> bool:
    print(f"waiting for {host} (ssh, up to {wait}s) ...")
    if not wait_for_ssh(host, wait):
        print(f"  {host} not reachable — is the Pi booted and the USB cable a data cable?")
        return False
    data = ssh_capture(cfg, host, COLLECT_CMD)
    if data is None:
        return False

    dup = next(
        (r["hostname"] for r in rows.values()
         if r.get("board_serial") and r["board_serial"] == data.get("board_serial")
         and r["hostname"] != hostname_key),
        None,
    )
    if dup:
        print(f"  WARNING: this board (serial {data['board_serial']}) was already "
              f"collected as {dup} — did you forget to swap Pis?")

    power = describe_throttled(data.get("throttled", ""))
    upsert(
        rows, hostname_key,
        index=index if index is not None else None,
        collected_at=now(),
        board_serial=data.get("board_serial", ""),
        model=data.get("model", ""),
        wlan0_mac=data.get("wlan0_mac", ""),
        eth0_mac=data.get("eth0_mac", ""),
        throttled=data.get("throttled", ""),
    )
    write_inventory(inventory, rows)
    print(f"  {hostname_key}: wlan0 {data.get('wlan0_mac', '?')}  "
          f"(board {data.get('board_serial', '?')}, reports hostname={data.get('hostname', '?')})")
    if power:
        print(f"  POWER WARNING: {power} — check the USB power source/cable")
    return True


def cmd_collect(args, cfg: dict) -> None:
    rows = read_inventory(args.inventory)

    if args.loop:
        host = args.host or die("--loop needs --host (the probe card's name, e.g. probe.local)")
        index = args.start
        while index <= cfg["fleet"]["count"]:
            hostname = hostname_for(cfg, index)
            answer = input(
                f"\n[{index}/{cfg['fleet']['count']}] connect the next Pi (will be {hostname}) "
                "and press Enter (q to quit): "
            ).strip().lower()
            if answer == "q":
                break
            if collect_one(cfg, rows, host, hostname, index, args.inventory, args.wait):
                print(f"  >>> label this board '{hostname}', then unplug it")
                index += 1
        print("\ncollected so far:")
        cmd_maclist(args, cfg)
        return

    if args.index is not None:
        hostname = hostname_for(cfg, args.index)
        host = args.host or f"{hostname}.local"
        collect_one(cfg, rows, host, hostname, args.index, args.inventory, args.wait)
    elif args.host:
        hostname = args.host.removesuffix(".local")
        collect_one(cfg, rows, args.host, hostname, None, args.inventory, args.wait)
    else:
        die("pass --index N (provisioned card) or --host NAME.local, or --loop for the probe-card sweep")


# --------------------------------------------------------------- verify


def cmd_verify(args, cfg: dict) -> None:
    rows = read_inventory(args.inventory)
    if args.index is not None:
        targets = [hostname_for(cfg, args.index)]
    else:
        targets = [h for h, r in sorted(rows.items()) if r.get("provisioned_at")]
    if not targets:
        die("nothing to verify — no provisioned entries in the inventory")

    ok = 0
    for hostname in targets:
        host = f"{hostname}.local"
        print(f"\n{hostname}:")
        if not wait_for_ssh(host, args.wait):
            print("  UNREACHABLE")
            continue
        data = ssh_capture(cfg, host, VERIFY_CMD)
        if data is None:
            continue
        problems = []
        if data.get("hostname") != hostname:
            problems.append(f"hostname mismatch (Pi says '{data.get('hostname')}')")
        if not data.get("wifi_ip"):
            problems.append("no IP on wlan0")
        if data.get("internet") != "ok":
            problems.append("no internet access")
        if args.camera and data.get("camera") != "ok":
            problems.append("no camera detected")
        power = describe_throttled(data.get("throttled", ""))
        if power:
            problems.append(f"power: {power}")

        print(f"  wifi: ssid={data.get('ssid') or '-'} ip={data.get('wifi_ip') or '-'}  "
              f"internet={data.get('internet')}  camera={data.get('camera')}")
        if problems:
            print("  PROBLEMS: " + "; ".join(problems))
            upsert(rows, hostname, notes="; ".join(problems))
        else:
            print("  OK")
            ok += 1
            upsert(rows, hostname, verified_at=now(),
                   wifi_ip=data.get("wifi_ip", ""), notes="")
        write_inventory(args.inventory, rows)

    print(f"\n{ok}/{len(targets)} verified OK")


# -------------------------------------------------------------- maclist


def cmd_maclist(args, cfg: dict) -> None:
    rows = read_inventory(args.inventory)
    collected = [r for _, r in sorted(rows.items()) if r.get("wlan0_mac")]
    if not collected:
        print("no MAC addresses collected yet")
        return
    print(f"{'hostname':<10} {'wlan0 MAC':<20} board serial")
    for r in collected:
        print(f"{r['hostname']:<10} {r['wlan0_mac']:<20} {r['board_serial']}")
    print(f"\n{len(collected)} MACs; bare list for the registration form:")
    for r in collected:
        print(r["wlan0_mac"])


# ----------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sdprov.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY,
                        help=f"inventory CSV (default: {DEFAULT_INVENTORY})")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("provision", help="customize the inserted SD card")
    p.add_argument("--volume", help="boot partition mount point (default: autodetect under /Volumes)")
    p.add_argument("--hostname", help="use this hostname instead of allocating the next piNN "
                                      "(e.g. 'probe' for the MAC-sweep card)")
    p.add_argument("--force", action="store_true",
                   help="allocate a new hostname even if the card was provisioned before")
    p.add_argument("--no-eject", dest="eject", action="store_false",
                   help="leave the card mounted after provisioning")
    p.set_defaults(func=cmd_provision)

    p = sub.add_parser("collect", help="record a booted Pi's serial + MACs in the inventory")
    p.add_argument("--index", type=int, help="fleet index of this Pi (fills the piNN row)")
    p.add_argument("--host", help="ssh host (default: piNN.local from --index)")
    p.add_argument("--loop", action="store_true",
                   help="probe-card sweep: prompt/collect/label for each Pi in turn")
    p.add_argument("--start", type=int, default=1, help="first index for --loop (default 1)")
    p.add_argument("--wait", type=int, default=180,
                   help="seconds to wait for ssh to come up (default 180)")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("verify", help="check hostname/WiFi/internet on provisioned Pis")
    p.add_argument("--index", type=int, help="verify a single Pi instead of all provisioned ones")
    p.add_argument("--camera", action="store_true", help="also require a detected camera")
    p.add_argument("--wait", type=int, default=20,
                   help="seconds to wait for each Pi's ssh (default 20)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("maclist", help="print collected WiFi MACs for registration")
    p.set_defaults(func=cmd_maclist)

    args = parser.parse_args()
    cfg = load_config(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
