#ifdef USE_ESP32

#include "audio_capture.h"

#include "esphome/core/application.h"
#include "esphome/core/log.h"

#include <esp_heap_caps.h>
#include <esp_crt_bundle.h>
#include <esp_http_client.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace esphome {
namespace audio_capture {

static const char *const TAG = "audio_capture";
static const size_t WAV_HEADER_SIZE = 44;
static const size_t UPLOAD_CHUNK = 4096;

// Raw bytes of a microphone chunk, whichever type this ESPHome version passes.
static inline void capture_bytes(const std::vector<uint8_t> &chunk, const uint8_t *&data, size_t &len) {
  data = chunk.data();
  len = chunk.size();
}
template<typename C>
static inline auto capture_bytes(const C &chunk, const uint8_t *&data, size_t &len)
    -> decltype((void) chunk.data.data(), void()) {
  data = chunk.data.data();
  len = chunk.data.size();
}

void AudioCapture::setup() {
  this->sample_rate_ = this->microphone_source_->get_audio_stream_info().get_sample_rate();
  // 16-bit mono; keep the size a whole number of samples.
  this->ring_size_ = (size_t) this->sample_rate_ * 2 * this->buffer_duration_ms_ / 1000;
  this->ring_size_ &= ~(size_t) 1;

  this->ring_ = (uint8_t *) heap_caps_malloc(this->ring_size_, MALLOC_CAP_SPIRAM);
  this->snapshot_ = (uint8_t *) heap_caps_malloc(WAV_HEADER_SIZE + this->ring_size_, MALLOC_CAP_SPIRAM);
  this->lock_ = xSemaphoreCreateMutex();
  this->frame_samples_ = this->sample_rate_ / 10;
  this->frame_count_ = this->ring_size_ / 2 / this->frame_samples_;
  this->frames_ = (uint8_t *) heap_caps_calloc(this->frame_count_, 1, MALLOC_CAP_SPIRAM);
  // Mean square of a full-scale 16-bit signal is 32768^2; scale by the dBFS threshold.
  this->frame_threshold_ = (uint64_t) (1073741824.0 * pow(10.0, this->ambient_min_level_db_ / 10.0));
  this->slot_start_ms_ = millis();
  if (this->ambient_per_day_ > 0) {
    // The first slot starts at boot, with its sample at a random moment like any other.
    this->slot_offset_ms_ = random_uint32() % (86400000UL / this->ambient_per_day_);
    this->slot_done_ = false;
  }
  if (this->ring_ == nullptr || this->snapshot_ == nullptr || this->lock_ == nullptr || this->frames_ == nullptr) {
    ESP_LOGE(TAG, "Could not allocate %u bytes of PSRAM", (unsigned) (2 * this->ring_size_));
    this->mark_failed();
    return;
  }

  // Runs on the microphone task. Never block it: if upload() holds the lock
  // while it copies the buffer out, this chunk (tens of ms) is dropped.
  // Generic so it builds on both ESPHome APIs: releases up to 2026.10 pass a
  // std::vector<uint8_t>, dev passes a microphone::AudioCapture.
  this->microphone_source_->add_data_callback([this](const auto &chunk) {
    if (xSemaphoreTake(this->lock_, 0) != pdTRUE) {
      return;
    }
    const uint8_t *data;
    size_t len;
    capture_bytes(chunk, data, len);
    if (len > this->ring_size_) {
      data += len - this->ring_size_;
      len = this->ring_size_;
    }
    const size_t first = std::min(len, this->ring_size_ - this->write_pos_);
    memcpy(this->ring_ + this->write_pos_, data, first);
    memcpy(this->ring_, data + first, len - first);
    this->write_pos_ = (this->write_pos_ + len) % this->ring_size_;
    this->filled_ = std::min(this->filled_ + len, this->ring_size_);
    xSemaphoreGive(this->lock_);

    // Track how much of the buffer has sound in it, 100 ms at a time.
    const int16_t *samples = reinterpret_cast<const int16_t *>(data);
    for (size_t i = 0; i < len / 2; i++) {
      const int32_t v = samples[i];
      this->frame_sum_sq_ += (uint64_t) (v * v);
      if (++this->frame_n_ >= this->frame_samples_) {
        this->frames_[this->frame_pos_] = (this->frame_sum_sq_ / this->frame_n_) > this->frame_threshold_;
        this->frame_pos_ = (this->frame_pos_ + 1) % this->frame_count_;
        this->frame_sum_sq_ = 0;
        this->frame_n_ = 0;
      }
    }
  });
}

void AudioCapture::dump_config() {
  ESP_LOGCONFIG(TAG,
                "Audio Capture:\n"
                "  Buffer: %u ms (%u bytes)\n"
                "  URL: %s",
                (unsigned) this->buffer_duration_ms_, (unsigned) this->ring_size_, this->url_.c_str());
  if (this->ambient_per_day_ > 0) {
    ESP_LOGCONFIG(TAG, "  Ambient samples: up to %u per day as '%s', needs %.0f%% of the buffer above %.0f dBFS",
                  this->ambient_per_day_, this->ambient_label_.c_str(), this->ambient_min_active_ * 100.0f,
                  this->ambient_min_level_db_);
  }
}

float AudioCapture::active_fraction_() const {
  size_t active = 0;
  for (size_t i = 0; i < this->frame_count_; i++) {
    active += this->frames_[i];
  }
  return (float) active / (float) this->frame_count_;
}

void AudioCapture::check_ambient_() {
  if (this->ambient_per_day_ == 0) {
    return;
  }
  const uint32_t now = millis();
  const uint32_t slot_ms = 86400000UL / this->ambient_per_day_;
  if (this->slot_done_ || now - this->slot_start_ms_ >= slot_ms) {
    if (now - this->slot_start_ms_ < slot_ms) {
      return;  // this slot's sample is taken (or was given up); wait for the next slot
    }
    // New slot: one chance, at a random moment inside it. A slot with no sound
    // after that moment is simply skipped -- nothing carries over.
    this->slot_start_ms_ += slot_ms * ((now - this->slot_start_ms_) / slot_ms);
    this->slot_offset_ms_ = random_uint32() % slot_ms;
    this->slot_done_ = false;
  }
  if (now - this->slot_start_ms_ < this->slot_offset_ms_ || now - this->last_ambient_check_ms_ < 2000) {
    return;
  }
  this->last_ambient_check_ms_ = now;
  // Never sample a buffer that holds a detection, a tagged miss, or anything
  // close to one: those are not known negatives.
  if (this->uploading_ || this->filled_ < this->ring_size_ ||
      (this->had_event_ && now - this->last_event_ms_ < this->buffer_duration_ms_ + 5000)) {
    return;
  }
  const float active = this->active_fraction_();
  if (active < this->ambient_min_active_) {
    // Checked every 2 s while a sample is due; log at most once a minute.
    if (now - this->last_skip_log_ms_ >= 60000) {
      this->last_skip_log_ms_ = now;
      ESP_LOGD(TAG, "Ambient sample due, but only %.0f%% of the buffer has sound (needs %.0f%%)", active * 100.0f,
               this->ambient_min_active_ * 100.0f);
    }
    return;
  }
  ESP_LOGD(TAG, "Ambient sample: %.0f%% of the buffer has sound", active * 100.0f);
  if (this->upload_(this->ambient_label_, true)) {
    this->slot_done_ = true;
  }
}

void AudioCapture::loop() {
  // Fire the automations from the main loop, not from the upload task.
  this->check_ambient_();
  const Result result = this->result_.exchange(NONE);
  if (result == OK) {
    this->uploaded_trigger_.trigger();
  } else if (result == FAILED) {
    this->error_trigger_.trigger();
  }
}

static void put_le(uint8_t *p, uint32_t value, size_t bytes) {
  for (size_t i = 0; i < bytes; i++) {
    p[i] = (value >> (8 * i)) & 0xFF;
  }
}

bool AudioCapture::upload_(const std::string &label, bool ambient) {
  if (this->is_failed()) {
    return false;
  }
  if (!ambient) {
    this->last_event_ms_ = millis();
    this->had_event_ = true;
  }
  if (this->uploading_.exchange(true)) {
    ESP_LOGW(TAG, "Upload already in progress, ignoring '%s'", label.c_str());
    return false;
  }

  // Copy the ring out oldest-first, behind a WAV header.
  xSemaphoreTake(this->lock_, portMAX_DELAY);
  const size_t filled = this->filled_;
  const size_t start = (this->write_pos_ + this->ring_size_ - filled) % this->ring_size_;
  const size_t first = std::min(filled, this->ring_size_ - start);
  uint8_t *pcm = this->snapshot_ + WAV_HEADER_SIZE;
  memcpy(pcm, this->ring_ + start, first);
  memcpy(pcm + first, this->ring_, filled - first);
  xSemaphoreGive(this->lock_);

  if (filled == 0) {
    ESP_LOGW(TAG, "Nothing recorded yet (is the microphone running?)");
    this->uploading_ = false;
    return false;
  }

  uint8_t *h = this->snapshot_;
  memcpy(h, "RIFF", 4);
  put_le(h + 4, 36 + filled, 4);
  memcpy(h + 8, "WAVEfmt ", 8);
  put_le(h + 16, 16, 4);                     // fmt chunk size
  put_le(h + 20, 1, 2);                      // PCM
  put_le(h + 22, 1, 2);                      // mono
  put_le(h + 24, this->sample_rate_, 4);
  put_le(h + 28, this->sample_rate_ * 2, 4);  // byte rate
  put_le(h + 32, 2, 2);                      // block align
  put_le(h + 34, 16, 2);                     // bits per sample
  memcpy(h + 36, "data", 4);
  put_le(h + 40, filled, 4);
  this->snapshot_size_ = WAV_HEADER_SIZE + filled;
  this->label_ = label;
  this->relabel_only_ = false;

  ESP_LOGI(TAG, "Uploading %u ms tagged '%s'", (unsigned) (filled / 2 * 1000 / this->sample_rate_), label.c_str());
  return this->start_task_();
}

bool AudioCapture::tag_last(const std::string &label) {
  if (this->is_failed()) {
    return false;
  }
  this->last_event_ms_ = millis();
  this->had_event_ = true;
  if (this->uploading_.exchange(true)) {
    ESP_LOGW(TAG, "Upload already in progress, ignoring re-tag '%s'", label.c_str());
    return false;
  }
  this->snapshot_size_ = 0;
  this->label_ = label;
  this->relabel_only_ = true;
  ESP_LOGI(TAG, "Re-tagging the last upload as '%s'", label.c_str());
  return this->start_task_();
}

bool AudioCapture::start_task_() {
  if (xTaskCreate(AudioCapture::upload_task, "audio_capture", 6144, this, 1, nullptr) != pdPASS) {
    ESP_LOGE(TAG, "Could not start the upload task");
    this->uploading_ = false;
    this->result_ = FAILED;
    return false;
  }
  return true;
}

void AudioCapture::upload_task(void *params) {
  auto *self = static_cast<AudioCapture *>(params);
  self->result_ = self->post_snapshot_() ? OK : FAILED;
  self->uploading_ = false;
  vTaskDelete(nullptr);
}

bool AudioCapture::post_snapshot_() {
  esp_http_client_config_t config = {};
  config.url = this->url_.c_str();
  config.method = HTTP_METHOD_POST;
  config.timeout_ms = 10000;
  // https:// URLs are verified against the ESP-IDF certificate bundle; it is
  // ignored for plain http://.
  config.crt_bundle_attach = esp_crt_bundle_attach;
  esp_http_client_handle_t client = esp_http_client_init(&config);
  if (client == nullptr) {
    ESP_LOGE(TAG, "Could not create the HTTP client");
    return false;
  }
  esp_http_client_set_header(client, "Content-Type", "audio/wav");
  esp_http_client_set_header(client, this->relabel_only_ ? "X-Relabel-Last" : "X-Label", this->label_.c_str());
  esp_http_client_set_header(client, "X-Device", App.get_name().c_str());
  if (!this->token_.empty()) {
    esp_http_client_set_header(client, "X-Capture-Token", this->token_.c_str());
  }

  bool ok = false;
  esp_err_t err = esp_http_client_open(client, this->snapshot_size_);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Could not connect to %s: %s", this->url_.c_str(), esp_err_to_name(err));
  } else {
    size_t sent = 0;
    while (sent < this->snapshot_size_) {
      const int n = esp_http_client_write(client, (const char *) this->snapshot_ + sent,
                                          std::min(UPLOAD_CHUNK, this->snapshot_size_ - sent));
      if (n <= 0) {
        break;
      }
      sent += n;
    }
    if (sent != this->snapshot_size_) {
      ESP_LOGE(TAG, "Upload stopped after %u of %u bytes", (unsigned) sent, (unsigned) this->snapshot_size_);
    } else {
      esp_http_client_fetch_headers(client);
      const int status = esp_http_client_get_status_code(client);
      ok = status >= 200 && status < 300;
      if (ok) {
        ESP_LOGI(TAG, "Uploaded %u bytes", (unsigned) sent);
      } else {
        ESP_LOGE(TAG, "Server answered HTTP %d", status);
      }
    }
  }
  esp_http_client_close(client);
  esp_http_client_cleanup(client);
  return ok;
}

}  // namespace audio_capture
}  // namespace esphome

#endif  // USE_ESP32
