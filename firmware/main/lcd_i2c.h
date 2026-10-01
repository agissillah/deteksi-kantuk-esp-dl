/*
 * lcd_i2c.h
 * ---------
 * Driver sederhana LCD 16x2 (HD44780) lewat modul I2C PCF8574.
 *
 * Pemasangan:
 *   GND -> GND,  VCC -> VIN/5V,  SDA -> GPIO 21,  SCL -> GPIO 22
 */

#pragma once

/* Menyiapkan I2C dan LCD. Alamat 0x27 / 0x3F dicari otomatis.
 * Mengembalikan true kalau LCD ditemukan. */
bool lcd_mulai(int pin_sda, int pin_scl);

/* Menulis satu baris (0 = atas, 1 = bawah). Teks dipotong/diisi spasi
 * sampai 16 karakter. Kalau isinya sama dengan sebelumnya, tidak ditulis ulang. */
void lcd_tulis_baris(int baris, const char *teks);
