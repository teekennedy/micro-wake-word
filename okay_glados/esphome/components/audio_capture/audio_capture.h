#pragma once

#ifdef USE_ESP32

#include "esphome/components/microphone/microphone_source.h"
#include "esphome/core/automation.h"
#include "esphome/core/component.h"

#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include <atomic>
#include <string>

namespace esphome {
namespace audio_capture {

/// Keeps the most recent `buffer_duration` of 16-bit mono audio in PSRAM and,
/// on request, POSTs a snapshot of it as a WAV file.
class AudioCapture : public Component {
 public:
  void setup() override;
  void loop() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::LATE; }

  void set_microphone_source(microphone::MicrophoneSource *source) { this->microphone_source_ = source; }
  void set_url(const std::string &url) { this->url_ = url; }
  void set_token(const std::string &token) { this->token_ = token; }
  void set_buffer_duration_ms(uint32_t ms) { this->buffer_duration_ms_ = ms; }

  /// Snapshot the buffer and upload it in the background. Returns false if an
  /// upload is already running or nothing has been recorded yet.
  bool upload(const std::string &label) { return this->upload_(label, false); }

  /// Occasionally upload a buffer that triggered nothing, as a known negative.
  void set_ambient(uint8_t per_day, float min_level_db, float min_active, const std::string &label) {
    this->ambient_per_day_ = per_day;
    this->ambient_min_level_db_ = min_level_db;
    this->ambient_min_active_ = min_active;
    this->ambient_label_ = label;
  }

  /// Ask the server to move this device's most recent upload to `label`
  /// (no audio is sent). Used to re-tag a "detected" clip as a false accept.
  bool tag_last(const std::string &label);

  Trigger<> *get_uploaded_trigger() { return &this->uploaded_trigger_; }
  Trigger<> *get_error_trigger() { return &this->error_trigger_; }

 protected:
  static void upload_task(void *params);
  bool post_snapshot_();
  bool start_task_();
  bool upload_(const std::string &label, bool ambient);
  void check_ambient_();
  float active_fraction_() const;

  microphone::MicrophoneSource *microphone_source_{nullptr};
  std::string url_;
  std::string token_;
  uint32_t buffer_duration_ms_{12000};
  uint32_t sample_rate_{16000};

  // Circular buffer, written from the microphone task.
  uint8_t *ring_{nullptr};
  size_t ring_size_{0};
  size_t write_pos_{0};
  size_t filled_{0};
  SemaphoreHandle_t lock_{nullptr};

  // Loudness of the buffer in 100 ms frames (1 = above ambient_min_level),
  // filled in by the microphone task alongside the ring.
  uint8_t *frames_{nullptr};
  size_t frame_count_{0};
  size_t frame_pos_{0};
  uint32_t frame_samples_{1600};
  uint64_t frame_sum_sq_{0};
  uint32_t frame_n_{0};
  uint64_t frame_threshold_{0};

  // Ambient sampling: the day is cut into `per_day` equal slots, and each slot
  // gets one randomly timed chance to upload a buffer that has sound in it.
  uint8_t ambient_per_day_{0};
  float ambient_min_level_db_{-78.0f};
  float ambient_min_active_{0.3f};
  std::string ambient_label_;
  uint32_t slot_start_ms_{0};
  uint32_t slot_offset_ms_{0};
  bool slot_done_{true};
  uint32_t last_ambient_check_ms_{0};
  uint32_t last_skip_log_ms_{0};
  uint32_t last_event_ms_{0};  // last detection/button upload or re-tag
  bool had_event_{false};

  // Linearised copy handed to the upload task.
  uint8_t *snapshot_{nullptr};
  size_t snapshot_size_{0};
  std::string label_;
  bool relabel_only_{false};

  enum Result : uint8_t { NONE, OK, FAILED };
  std::atomic<bool> uploading_{false};
  std::atomic<Result> result_{NONE};

  Trigger<> uploaded_trigger_;
  Trigger<> error_trigger_;
};

template<typename... Ts> class UploadAction : public Action<Ts...>, public Parented<AudioCapture> {
 public:
  TEMPLATABLE_VALUE(std::string, label)
  void play(const Ts &...x) override { this->parent_->upload(this->label_.value(x...)); }
};

template<typename... Ts> class TagLastAction : public Action<Ts...>, public Parented<AudioCapture> {
 public:
  TEMPLATABLE_VALUE(std::string, label)
  void play(const Ts &...x) override { this->parent_->tag_last(this->label_.value(x...)); }
};

}  // namespace audio_capture
}  // namespace esphome

#endif  // USE_ESP32
