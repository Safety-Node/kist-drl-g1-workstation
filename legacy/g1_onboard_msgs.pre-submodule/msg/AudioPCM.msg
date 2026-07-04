# AudioPCM — raw PCM audio chunk between sensors/{mic,speaker}_node and comm_bridge.
#
# Pipeline format LOCKED:
#   sample_rate = 16000  (Hz)
#   channels    = 1      (mono)
#   bit_depth   = 16     (signed little-endian)
#
# Publishers with different native rates (e.g. CLOVA TTS at 24 kHz) MUST resample
# to the locked format before publishing. The fields below let a future "mixed rate"
# path opt in without re-spinning the wire format, but current consumers
# (speaker_node, STT Provider) assume the locked values.

std_msgs/Header header
uint32 sample_rate           # Hz (pipeline: 16000)
uint8  channels              # channel count (pipeline: 1)
uint8  bit_depth             # bits per sample (pipeline: 16)
uint8[] data                 # interleaved PCM, signed int16 LE packed into bytes
