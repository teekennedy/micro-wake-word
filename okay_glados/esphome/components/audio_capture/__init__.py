"""audio_capture -- keep the last N seconds of microphone audio and upload it on demand.

Built to collect wake word training data from the device that will run the
model: tag the buffered audio as a false accept / false reject with a
button press and it is POSTed as a WAV file to a server on the local network.
"""

from esphome import automation
import esphome.codegen as cg
from esphome.components import microphone
import esphome.config_validation as cv
from esphome.const import CONF_ID, CONF_MICROPHONE, CONF_URL

CODEOWNERS = []
DEPENDENCIES = ["microphone", "network", "psram"]

CONF_BUFFER_DURATION = "buffer_duration"
CONF_LABEL = "label"
CONF_AMBIENT_SAMPLES = "ambient_samples"
CONF_PER_DAY = "per_day"
CONF_MIN_LEVEL = "min_level"
CONF_MIN_ACTIVE = "min_active"
CONF_ON_UPLOADED = "on_uploaded"
CONF_ON_ERROR = "on_error"

audio_capture_ns = cg.esphome_ns.namespace("audio_capture")
AudioCapture = audio_capture_ns.class_("AudioCapture", cg.Component)
UploadAction = audio_capture_ns.class_("UploadAction", automation.Action)
TagLastAction = audio_capture_ns.class_("TagLastAction", automation.Action)

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(AudioCapture),
        # Use the same channel and gain_factor as micro_wake_word so the
        # saved audio is exactly what the model heard.
        cv.Required(CONF_MICROPHONE): microphone.microphone_source_schema(
            min_bits_per_sample=16, max_bits_per_sample=16, min_channels=1, max_channels=1
        ),
        cv.Required(CONF_URL): cv.url,
        cv.Optional(CONF_BUFFER_DURATION, default="12s"): cv.All(
            cv.positive_time_period_milliseconds,
            cv.Range(min=cv.TimePeriod(seconds=1), max=cv.TimePeriod(seconds=60)),
        ),
        # Now and then, upload a buffer that triggered nothing but has sound in
        # it: real negatives from the device's own microphone. The day is cut
        # into `per_day` equal slots with one randomly timed sample each, so
        # the samples spread over the day and can never exceed `per_day`.
        cv.Optional(CONF_AMBIENT_SAMPLES): cv.Schema(
            {
                cv.Optional(CONF_PER_DAY, default=8): cv.int_range(min=1, max=96),
                cv.Optional(CONF_LABEL, default="true_reject"): cv.string,
                # A 100 ms frame counts as "sound" above this level (dBFS RMS)...
                cv.Optional(CONF_MIN_LEVEL, default=-50.0): cv.float_range(min=-90.0, max=0.0),
                # ...and this much of the buffer must be sound.
                cv.Optional(CONF_MIN_ACTIVE, default="30%"): cv.percentage,
            }
        ),
        cv.Optional(CONF_ON_UPLOADED): automation.validate_automation(single=True),
        cv.Optional(CONF_ON_ERROR): automation.validate_automation(single=True),
    }
).extend(cv.COMPONENT_SCHEMA)

FINAL_VALIDATE_SCHEMA = cv.Schema(
    {
        cv.Required(CONF_MICROPHONE): microphone.final_validate_microphone_source_schema(
            "audio_capture", sample_rate=16000
        ),
    },
    extra=cv.ALLOW_EXTRA,
)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    # Passive: never starts or stops the microphone, only listens while
    # micro_wake_word has it running.
    mic_source = await microphone.microphone_source_to_code(config[CONF_MICROPHONE], passive=True)
    cg.add(var.set_microphone_source(mic_source))
    cg.add(var.set_url(config[CONF_URL]))
    cg.add(var.set_buffer_duration_ms(config[CONF_BUFFER_DURATION]))
    if ambient := config.get(CONF_AMBIENT_SAMPLES):
        cg.add(
            var.set_ambient(
                ambient[CONF_PER_DAY],
                ambient[CONF_MIN_LEVEL],
                ambient[CONF_MIN_ACTIVE],
                ambient[CONF_LABEL],
            )
        )
    if CONF_ON_UPLOADED in config:
        await automation.build_automation(var.get_uploaded_trigger(), [], config[CONF_ON_UPLOADED])
    if CONF_ON_ERROR in config:
        await automation.build_automation(var.get_error_trigger(), [], config[CONF_ON_ERROR])


_LABEL_ACTION_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.use_id(AudioCapture),
        cv.Required(CONF_LABEL): cv.templatable(cv.string),
    }
)


@automation.register_action(
    "audio_capture.upload", UploadAction, _LABEL_ACTION_SCHEMA, synchronous=True
)
@automation.register_action(
    "audio_capture.tag_last", TagLastAction, _LABEL_ACTION_SCHEMA, synchronous=True
)
async def upload_action_to_code(config, action_id, template_arg, args):
    var = cg.new_Pvariable(action_id, template_arg)
    await cg.register_parented(var, config[CONF_ID])
    cg.add(var.set_label(await cg.templatable(config[CONF_LABEL], args, cg.std_string)))
    return var
