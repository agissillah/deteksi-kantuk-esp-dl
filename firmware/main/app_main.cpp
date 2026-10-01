/*
 * app_main.cpp
 * ------------
 * Firmware ESP32 untuk deteksi kantuk.
 *
 * Tugas ESP32:
 *   1. Menerima potongan gambar mata (32x32 grayscale) dari laptop lewat serial
 *   2. Menjalankan inference memakai ESP-DL (model eyenet.espdl, INT8)
 *   3. Menghitung berapa lama mata tertutup terus-menerus
 *   4. Membunyikan buzzer di GPIO 25 kalau melewati batas durasi
 *   5. Mengirim hasilnya kembali ke laptop
 *
 * Format paket dari laptop:
 *   0xA5 0x5A  <jumlah_mata>  <data mata 1 (1024 byte)> [<data mata 2 (1024 byte)>]
 *   jumlah_mata = 1 atau 2, tiap mata 32x32 = 1024 piksel grayscale (0-255)
 *
 * Balasan ke laptop (satu baris teks):
 *   RES <OPEN|CLOSED> <peluang_tertutup> <lama_tertutup_ms> <buzzer 0|1> <waktu_inference_us>
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_log.h"

#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

#include "lcd_i2c.h"

static const char *TAG = "KANTUK";

// ---------------- pengaturan ----------------

#define PIN_BUZZER       GPIO_NUM_25    // buzzer pada GPIO 25
#define NADA_BUZZER      4000           // frekuensi bunyi buzzer (Hz), hasil uji: buzzer ini hanya bunyi di 4000 Hz

// Cara menyalakan buzzer:
//   0 = tegangan DC terus  -> untuk buzzer AKTIF
//   1 = gelombang PWM      -> untuk buzzer PASIF
// Jalankan uji buzzer (kirim huruf T lewat serial) untuk tahu mana yang cocok.
#define MODE_BUZZER      1

#define PIN_LCD_SDA      21             // LCD 16x2 I2C: SDA
#define PIN_LCD_SCL      22             // LCD 16x2 I2C: SCL
#define JEDA_LCD_MS      250            // LCD diperbarui paling cepat tiap 250 ms

#define UKURAN_GAMBAR    32
#define PIKSEL_PER_MATA  (UKURAN_GAMBAR * UKURAN_GAMBAR)   // 1024
#define MAX_MATA         2

#define AMBANG_TERTUTUP  0.5f           // peluang di atas ini dianggap mata tertutup
#define DURASI_KANTUK_MS 1500           // mata tertutup selama ini -> buzzer menyala

#define UART_PORT        UART_NUM_0
#define UART_BAUD        921600
#define BUF_UART         4096

// urutan keluaran model: indeks 0 = CLOSED, 1 = OPEN
#define IDX_CLOSED 0
#define IDX_OPEN   1

// model ditanam langsung di dalam firmware
extern const uint8_t eyenet_espdl[] asm("_binary_eyenet_espdl_start");

static dl::Model *model = nullptr;
static dl::TensorBase *tensor_masuk = nullptr;
static dl::TensorBase *tensor_keluar = nullptr;
static int exp_masuk = 0;      // faktor skala kuantisasi masukan
static int exp_keluar = 0;     // faktor skala kuantisasi keluaran

// ---------------- buzzer ----------------

static void buzzer_siapkan();   // deklarasi maju

static void buzzer_siapkan()
{
    // LEDC dipakai supaya buzzer pasif ikut berbunyi (buzzer aktif juga tetap bunyi)
    ledc_timer_config_t timer = {};
    timer.speed_mode = LEDC_LOW_SPEED_MODE;
    timer.duty_resolution = LEDC_TIMER_10_BIT;
    timer.timer_num = LEDC_TIMER_0;
    timer.freq_hz = NADA_BUZZER;
    timer.clk_cfg = LEDC_AUTO_CLK;
    ledc_timer_config(&timer);

    ledc_channel_config_t kanal = {};
    kanal.gpio_num = PIN_BUZZER;
    kanal.speed_mode = LEDC_LOW_SPEED_MODE;
    kanal.channel = LEDC_CHANNEL_0;
    kanal.timer_sel = LEDC_TIMER_0;
    kanal.duty = 0;                      // mulai dalam keadaan diam
    kanal.hpoint = 0;
    ledc_channel_config(&kanal);

    // arus keluaran pin dinaikkan ke tingkat maksimum (±40 mA) supaya buzzer lebih keras
    gpio_set_drive_capability(PIN_BUZZER, GPIO_DRIVE_CAP_3);
}

static void buzzer_set(bool nyala)
{
    if (MODE_BUZZER == 0) {
        // buzzer AKTIF: cukup diberi tegangan tinggi terus-menerus
        ledc_stop(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, nyala ? 1 : 0);
    } else {
        // buzzer PASIF: perlu gelombang kotak (PWM), duty 50%
        ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, nyala ? 512 : 0);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
    }
}

/*
 * Mencari frekuensi yang paling keras untuk buzzer pasif.
 * Tiap nada berbunyi 1 detik lalu diam 0,7 detik; dengarkan mana yang paling keras.
 * Jalankan dengan mengirim huruf 'T' lewat serial.
 */
static void uji_buzzer()
{
    const int nada[6] = {2500, 3000, 3500, 4000, 4500, 5000};
    for (int i = 0; i < 6; i++) {
        printf("NADA %d: %d Hz\n", i + 1, nada[i]);
        ledc_set_freq(LEDC_LOW_SPEED_MODE, LEDC_TIMER_0, nada[i]);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, 512);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
        vTaskDelay(pdMS_TO_TICKS(1000));
        ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, 0);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
        vTaskDelay(pdMS_TO_TICKS(700));
    }

    printf("UJI SELESAI. Kembali ke %d Hz.\n", NADA_BUZZER);
    ledc_set_freq(LEDC_LOW_SPEED_MODE, LEDC_TIMER_0, NADA_BUZZER);
    buzzer_set(false);
}

// ---------------- model ----------------

static bool model_siapkan()
{
    model = new dl::Model((const char *)eyenet_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    if (model == nullptr) {
        ESP_LOGE(TAG, "gagal memuat model");
        return false;
    }

    tensor_masuk = model->get_inputs().begin()->second;
    tensor_keluar = model->get_outputs().begin()->second;

    // exponent bertipe dl::ExponentInfo, ambil nilai int-nya dengan get()
    exp_masuk = tensor_masuk->exponent.get();
    exp_keluar = tensor_keluar->exponent.get();

    ESP_LOGI(TAG, "model siap. input exponent=%d, output exponent=%d", exp_masuk, exp_keluar);
    return true;
}

/*
 * Menjalankan model untuk satu potongan mata.
 * piksel: 1024 byte grayscale (0-255)
 * Mengembalikan peluang mata TERTUTUP (0..1).
 */
static float klasifikasi_satu_mata(const uint8_t *piksel)
{
    int8_t *masuk = (int8_t *)tensor_masuk->data;

    // Normalisasi HARUS sama persis dengan saat training di laptop:
    //   x = (piksel / 255 - 0.5) / 0.5   ->  rentang -1 .. 1
    for (int i = 0; i < PIKSEL_PER_MATA; i++) {
        float nilai = ((float)piksel[i] / 255.0f - 0.5f) / 0.5f;
        masuk[i] = dl::quantize<int8_t>(nilai, DL_RESCALE(exp_masuk));
    }

    model->run();

    // keluaran model: 2 angka (logit) untuk CLOSED dan OPEN
    int8_t *keluar = (int8_t *)tensor_keluar->data;
    float logit_closed = dl::dequantize(keluar[IDX_CLOSED], DL_SCALE(exp_keluar));
    float logit_open = dl::dequantize(keluar[IDX_OPEN], DL_SCALE(exp_keluar));

    // softmax untuk 2 kelas
    float selisih = logit_open - logit_closed;
    return 1.0f / (1.0f + expf(selisih));     // peluang CLOSED
}

// ---------------- komunikasi serial ----------------

static void uart_siapkan()
{
    uart_config_t cfg = {};
    cfg.baud_rate = UART_BAUD;
    cfg.data_bits = UART_DATA_8_BITS;
    cfg.parity = UART_PARITY_DISABLE;
    cfg.stop_bits = UART_STOP_BITS_1;
    cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    cfg.source_clk = UART_SCLK_DEFAULT;

    uart_driver_install(UART_PORT, BUF_UART, BUF_UART, 0, NULL, 0);
    uart_param_config(UART_PORT, &cfg);
}

/* Membaca tepat n byte dari serial. Mengembalikan false kalau waktu habis. */
static bool baca_pasti(uint8_t *tujuan, int n, int timeout_ms)
{
    int sudah = 0;
    while (sudah < n) {
        int dapat = uart_read_bytes(UART_PORT, tujuan + sudah, n - sudah,
                                    pdMS_TO_TICKS(timeout_ms));
        if (dapat <= 0) {
            return false;
        }
        sudah += dapat;
    }
    return true;
}

/* Menunggu sampai ketemu urutan header 0xA5 0x5A. */
static bool tunggu_header()
{
    uint8_t b = 0;
    while (true) {
        if (!baca_pasti(&b, 1, 1000)) {
            return false;
        }
        if (b == 'T' || b == 't') {          // perintah uji buzzer
            uji_buzzer();
            continue;
        }
        if (b == 0xA5) {
            if (!baca_pasti(&b, 1, 100)) {
                return false;
            }
            if (b == 0x5A) {
                return true;
            }
        }
    }
}

// ---------------- program utama ----------------

extern "C" void app_main(void)
{
    buzzer_siapkan();
    buzzer_set(false);
    uart_siapkan();

    if (!model_siapkan()) {
        while (true) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    // Uji buzzer TIDAK dijalankan otomatis saat booting, karena akan berbunyi
    // tiap kali program di laptop dijalankan. Untuk mengujinya, kirim huruf 'T'
    // lewat serial.
    ESP_LOGI(TAG, "menunggu data dari laptop ...");

    // LCD boleh tidak terpasang; sistem tetap jalan tanpa LCD
    lcd_mulai(PIN_LCD_SDA, PIN_LCD_SCL);
    lcd_tulis_baris(0, "Deteksi Kantuk");
    lcd_tulis_baris(1, "Menunggu data..");

    static uint8_t penampung[MAX_MATA * PIKSEL_PER_MATA];
    int64_t mulai_tertutup_us = -1;      // -1 artinya mata sedang terbuka
    int64_t lcd_terakhir_us = 0;
    char balasan[128];
    char teks_lcd[20];

    while (true) {
        if (!tunggu_header()) {
            // 1 detik tanpa data dari laptop: matikan buzzer supaya tidak
            // berbunyi terus kalau program di laptop ditutup
            buzzer_set(false);
            mulai_tertutup_us = -1;
            lcd_tulis_baris(0, "Deteksi Kantuk");
            lcd_tulis_baris(1, "Menunggu data..");
            continue;
        }

        uint8_t jumlah_mata = 0;
        if (!baca_pasti(&jumlah_mata, 1, 200)) {
            continue;
        }
        if (jumlah_mata < 1 || jumlah_mata > MAX_MATA) {
            continue;
        }

        int jumlah_byte = jumlah_mata * PIKSEL_PER_MATA;
        if (!baca_pasti(penampung, jumlah_byte, 500)) {
            continue;
        }

        // ---- inference ----
        int64_t t0 = esp_timer_get_time();
        float total_peluang = 0.0f;
        for (int i = 0; i < jumlah_mata; i++) {
            total_peluang += klasifikasi_satu_mata(penampung + i * PIKSEL_PER_MATA);
        }
        float peluang_tertutup = total_peluang / jumlah_mata;
        int64_t lama_inference_us = esp_timer_get_time() - t0;

        bool tertutup = peluang_tertutup > AMBANG_TERTUTUP;

        // ---- hitung durasi mata tertutup ----
        int64_t sekarang = esp_timer_get_time();
        int64_t lama_tertutup_ms = 0;

        if (tertutup) {
            if (mulai_tertutup_us < 0) {
                mulai_tertutup_us = sekarang;
            }
            lama_tertutup_ms = (sekarang - mulai_tertutup_us) / 1000;
        } else {
            mulai_tertutup_us = -1;
        }

        bool mengantuk = lama_tertutup_ms >= DURASI_KANTUK_MS;
        buzzer_set(mengantuk);

        // ---- tampilkan di LCD (dibatasi supaya inference tidak melambat) ----
        if (mengantuk || sekarang - lcd_terakhir_us >= JEDA_LCD_MS * 1000) {
            lcd_terakhir_us = sekarang;
            lcd_tulis_baris(0, tertutup ? "Mata: TERTUTUP" : "Mata: TERBUKA");
            if (mengantuk) {
                lcd_tulis_baris(1, "!! MENGANTUK !!");
            } else {
                snprintf(teks_lcd, sizeof(teks_lcd), "Tutup: %.1f dtk",
                         lama_tertutup_ms / 1000.0f);
                lcd_tulis_baris(1, teks_lcd);
            }
        }

        // ---- kirim hasil ke laptop ----
        int n = snprintf(balasan, sizeof(balasan), "RES %s %.3f %lld %d %lld\n",
                         tertutup ? "CLOSED" : "OPEN",
                         peluang_tertutup,
                         lama_tertutup_ms,
                         mengantuk ? 1 : 0,
                         lama_inference_us);
        uart_write_bytes(UART_PORT, balasan, n);
    }
}
