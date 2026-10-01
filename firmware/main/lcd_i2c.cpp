/*
 * lcd_i2c.cpp
 * -----------
 * Driver sederhana LCD 16x2 (HD44780) lewat modul I2C PCF8574.
 *
 * Modul I2C di belakang LCD mengubah 1 byte I2C menjadi 8 pin keluaran:
 *   P0 = RS, P1 = RW, P2 = EN, P3 = lampu latar, P4..P7 = data D4..D7
 * Karena hanya 4 jalur data, setiap byte dikirim dalam 2 bagian (mode 4-bit).
 */

#include "lcd_i2c.h"

#include <stdio.h>
#include <string.h>

#include "driver/i2c_master.h"
#include "esp_rom_sys.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define LCD_KOLOM 16

#define BIT_RS 0x01     // 0 = perintah, 1 = karakter
#define BIT_EN 0x04     // pulsa "enable": data dibaca LCD saat EN turun
#define BIT_BL 0x08     // lampu latar menyala

static i2c_master_bus_handle_t bus = nullptr;
static i2c_master_dev_handle_t lcd = nullptr;
static char isi_sekarang[2][LCD_KOLOM + 1];

/* Mengirim 4 bit atas dari 'nibble' disertai pulsa EN. */
static void kirim_nibble(uint8_t nibble, uint8_t mode)
{
    uint8_t data = (nibble & 0xF0) | mode | BIT_BL;
    uint8_t buf[2] = {(uint8_t)(data | BIT_EN), data};   // EN naik lalu turun
    i2c_master_transmit(lcd, buf, sizeof(buf), 50);
}

static void kirim_byte(uint8_t nilai, uint8_t mode)
{
    kirim_nibble(nilai & 0xF0, mode);            // 4 bit atas dulu
    kirim_nibble((uint8_t)(nilai << 4), mode);   // lalu 4 bit bawah
}

static void perintah(uint8_t cmd)
{
    kirim_byte(cmd, 0);
    if (cmd == 0x01 || cmd == 0x02) {
        vTaskDelay(pdMS_TO_TICKS(2));            // clear & home butuh waktu > 1,5 ms
    }
}

bool lcd_mulai(int pin_sda, int pin_scl)
{
    i2c_master_bus_config_t cfg = {};
    cfg.i2c_port = I2C_NUM_0;
    cfg.sda_io_num = (gpio_num_t)pin_sda;
    cfg.scl_io_num = (gpio_num_t)pin_scl;
    cfg.clk_source = I2C_CLK_SRC_DEFAULT;
    cfg.glitch_ignore_cnt = 7;
    cfg.flags.enable_internal_pullup = 1;

    if (i2c_new_master_bus(&cfg, &bus) != ESP_OK) {
        printf("LCD: gagal menyiapkan I2C\n");
        return false;
    }

    // alamat modul I2C LCD biasanya 0x27 (PCF8574T) atau 0x3F (PCF8574AT)
    uint16_t alamat = 0;
    const uint16_t kandidat[2] = {0x27, 0x3F};
    for (int i = 0; i < 2; i++) {
        if (i2c_master_probe(bus, kandidat[i], 50) == ESP_OK) {
            alamat = kandidat[i];
            break;
        }
    }

    if (alamat == 0) {
        printf("LCD: tidak ditemukan di 0x27 / 0x3F. Perangkat I2C yang terdeteksi:");
        for (uint16_t a = 0x08; a < 0x78; a++) {
            if (i2c_master_probe(bus, a, 20) == ESP_OK) {
                printf(" 0x%02X", a);
            }
        }
        printf("\n");
        return false;
    }

    i2c_device_config_t dev = {};
    dev.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    dev.device_address = alamat;
    dev.scl_speed_hz = 100000;
    if (i2c_master_bus_add_device(bus, &dev, &lcd) != ESP_OK) {
        printf("LCD: gagal mendaftarkan perangkat 0x%02X\n", alamat);
        return false;
    }

    // urutan inisialisasi HD44780 mode 4-bit (sesuai datasheet)
    vTaskDelay(pdMS_TO_TICKS(50));
    kirim_nibble(0x30, 0);
    vTaskDelay(pdMS_TO_TICKS(5));
    kirim_nibble(0x30, 0);
    esp_rom_delay_us(150);
    kirim_nibble(0x30, 0);
    esp_rom_delay_us(150);
    kirim_nibble(0x20, 0);      // masuk mode 4-bit

    perintah(0x28);   // 4-bit, 2 baris, huruf 5x8
    perintah(0x08);   // layar mati sementara
    perintah(0x01);   // bersihkan layar
    perintah(0x06);   // kursor bergeser ke kanan setelah tiap huruf
    perintah(0x0C);   // layar menyala, kursor disembunyikan

    memset(isi_sekarang, 0, sizeof(isi_sekarang));
    printf("LCD: ditemukan di alamat 0x%02X\n", alamat);
    return true;
}

void lcd_tulis_baris(int baris, const char *teks)
{
    if (lcd == nullptr || baris < 0 || baris > 1) {
        return;
    }

    // potong/isi spasi sampai 16 karakter supaya sisa tulisan lama terhapus
    char buf[LCD_KOLOM + 1];
    bool habis = false;
    for (int i = 0; i < LCD_KOLOM; i++) {
        if (!habis && teks[i] == '\0') {
            habis = true;
        }
        buf[i] = habis ? ' ' : teks[i];
    }
    buf[LCD_KOLOM] = '\0';

    if (strcmp(buf, isi_sekarang[baris]) == 0) {
        return;                                  // sama dengan yang tampil, lewati
    }
    strcpy(isi_sekarang[baris], buf);

    perintah(0x80 | (baris == 0 ? 0x00 : 0x40)); // pindah ke awal baris
    for (int i = 0; i < LCD_KOLOM; i++) {
        kirim_byte((uint8_t)buf[i], BIT_RS);
    }
}
