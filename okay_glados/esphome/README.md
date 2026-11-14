# Wake word data capture for the Home Assistant Voice PE

Collects training data from the device that runs the model: the same microphone
channel and gain the wake word model hears.

**Status:** compiles against the Voice PE config (ESPHome at the commit the
Voice PE package pins). It has **not been run on a device yet**, so expect to
debug the first flash. The TLS and token support were added afterwards and
have not been compiled.

| File | What it is |
|---|---|
| `components/audio_capture/` | ESPHome external component: a ring buffer of recent audio and an uploader |
| `audio-capture.yaml` | Package that wires the component into the Voice PE: detections, "stop", buttons, ambient samples |
| `capture_server.py` | Minimal stand-in receiver (Python standard library only) |

## What gets saved

Every clip is the last 12 s of audio, as a 16 kHz mono WAV.

| Label | When |
|---|---|
| `detected` | Every time the wake word fires. Uploaded 300 ms after the detection. |
| `false_accept` | A `detected` clip, re-labelled because you said "stop" before the reply started, or triple-pressed the centre button. No audio is re-sent; the server moves the clip. |
| `false_reject` | You double-pressed the centre button: "I just said it and it ignored me". Press within about 8 s. |
| `true_reject` | Up to 8 times a day, a buffer that had sound in it and triggered nothing. |

The package also enables the built-in `stop` wake word while the assistant is
thinking or replying, so "stop" cancels a request. It only *tags* a false accept
if said before the reply starts; during a reply it just stops playback.

## 1. Receiver

The real receiver is the wake word trainer's collector
(`k8s-homelab`, `k8s/apps/wake-word-trainer`), at
`https://wake-word.msng.to/capture`. It needs the `X-Capture-Token` header;
read the token from the cluster and put it in the ESPHome `secrets.yaml`:

```sh
kubectl -n wake-word-trainer get secret wake-word-trainer-capture-token \
  -o jsonpath='{.data.token}' | base64 -d
```

```yaml
# secrets.yaml
wake_word_capture_token: "<token>"
```

For a quick local test, `capture_server.py` speaks the same contract over plain
HTTP and ignores the token:

```sh
python3 capture_server.py --dir /path/to/captures --port 8787
```

Clips land in `<dir>/<label>/<device>_<YYYYmmdd-HHMMSS>.wav`. An ambient sample
that is followed within 30 s by a detection or a button press is moved to
`discarded/`, since it probably contains the wake word. Point `capture_url` at
`http://<machine>:8787/` to use it.

The device must be allowed to reach the receiver. On a network that isolates
IoT devices this needs a firewall rule for the port.

## 2. Add the package to the device config

Include the package from the public GitHub mirror **after** the Voice PE
package. It loads the `audio_capture` component from the same repository, so
the ESPHome builder needs no local files or credentials:

```yaml
packages:
  Nabu Casa.Home Assistant Voice PE:
    url: https://github.com/esphome/home-assistant-voice-pe
    files: [home-assistant-voice.yaml]
    ref: dev
  capture:
    url: https://github.com/teekennedy/micro-wake-word
    ref: main
    files:
      - path: okay_glados/esphome/audio-capture.yaml
        vars:
          capture_token: !secret wake_word_capture_token
```

Variables: `capture_url` (default the collector), `capture_token` and
`capture_ref` (the git ref the component is fetched at; keep it equal to the
package `ref`).

Nothing else in your config needs to change. The package appends to the Voice
PE's own `on_wake_word_detected`, `voice_assistant` and button-event
automations rather than replacing them.

## 3. Check it works

Watch the device log (`esphome logs`) and the server output:

1. Say the wake word. The device logs `Uploading 12000 ms tagged 'detected'`
   and the server prints `saved .../detected/...wav`.
2. Trigger it, give a command, and say "stop" while the LED shows it thinking.
   The server prints `re-tagged ... -> .../false_accept/...`.
3. Double-press the centre button: a `false_reject` clip appears.
4. Listen to a saved clip. It should sound like the room, at a sensible level,
   with the wake word at the very end of a `detected` clip.

## Tuning

These are the `audio_capture:` options, set in `audio-capture.yaml`:

| Option | Default | Notes |
|---|---|---|
| `url` | required | Where clips are POSTed. `https://` URLs are verified against the ESP-IDF certificate bundle. |
| `token` | none | Sent as `X-Capture-Token`. |
| `buffer_duration` | `12s` | 1–60 s. Costs about 64 KB of PSRAM per second (two copies). |
| `microphone` | required | Keep `channels` and `gain_factor` equal to the `micro_wake_word` microphone in the Voice PE package, or the clips will not match what the model hears. |
| `ambient_samples.per_day` | `8` | Hard cap. The day is cut into this many equal slots, counted from boot, with one randomly timed sample each. Remove the `ambient_samples` block to turn sampling off. |
| `ambient_samples.min_level` | `-50` | dBFS. A 100 ms frame counts as sound above this. **Not calibrated against a real device**: raise it if quiet rooms get sampled, lower it if nothing ever is. |
| `ambient_samples.min_active` | `30%` | Share of the buffer that must be sound. |
| `on_uploaded`, `on_error` | none | Automations run after an upload succeeds or fails. |

Actions, usable in any automation:

```yaml
- audio_capture.upload:      # send the current buffer
    id: wake_word_capture
    label: false_reject
- audio_capture.tag_last:    # re-label this device's newest `detected` clip
    id: wake_word_capture
    label: false_accept
```

## Known limitations

- **No retry.** If the server is unreachable when a clip is sent, that clip is
  lost, and a later re-tag has nothing to move.
- **One upload at a time.** A second request while one is in flight is dropped
  with a warning in the log.
- **"Stop" is not active while the device is still listening**, only once the
  command has been sent off. A real command containing "stop" is not cut short.
- **A genuine request cancelled with "stop" is tagged `false_accept`.** Review
  that folder before training on it.
- **Clips nobody tags stay in `detected/`.** They are probably real wake words,
  but an unnoticed false accept would sit there too.

## HTTP contract

For anyone writing a different receiver:

- `POST <url>` with a WAV body and headers `X-Label: <label>`,
  `X-Device: <device name>` and, when configured, `X-Capture-Token`. Any 2xx
  is success.
- `POST <url>` with an empty body and `X-Relabel-Last: <label>`: move that
  device's most recent `detected` clip to `<label>`.
