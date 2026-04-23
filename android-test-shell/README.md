# Android Test Shell

Bu klasor, backend'i Android emulator uzerinden test etmek icin hazirlanmis kucuk bir Android uygulama iskeletidir.

## Amac

Tam mobil urun degil. Sadece:
- gorsel secmek
- opsiyonel WAV secmek
- backend'e WebSocket ile gondermek
- JSON sonucu gormek
- MP3 sonucu oynatmak

## Nasil Kullanilir

1. Android Studio ile `android-test-shell` klasorunu ac.
2. Gerekirse SDK/JDK 17 uyumunu tamamla.
3. Emulator baslat.
4. Backend'i host makinede `127.0.0.1:8000` veya `0.0.0.0:8000` uzerinde calistir.
5. App icindeki varsayilan URL:

```text
ws://10.0.2.2:8000/ws/vision
```

Bu adres Android Emulator icinden host makineyi gosterir.

Lokal test shell cleartext `ws://` baglantilarina izin verir.
Bu sadece emulator + local backend test akisi icindir.

## Test Akislari

### Scene testi
- Gorsel sec
- `Scene Gonder`

### STT + image testi
- Gorsel sec
- WAV sec
- `Audio+Image Gonder`

## Not

Canli mikrofon kaydi bu ilk test shell surumunde eklenmedi. Ilk hedef backend entegrasyonunu emulator icinde dogrulamak.

## Gradle Notu

Bu shell AGP 9 kullaniyor.

Bu nedenle:
- `org.jetbrains.kotlin.android` plugin'i ayrica uygulanmaz
- `android { kotlinOptions { ... } }` DSL'i de kullanilmaz
- Kotlin destegi AGP tarafinda built-in gelir
- `jvmTarget` ayrica verilmez; `compileOptions.targetCompatibility` uzerinden alinur
- `org.jetbrains.kotlin.plugin.compose` plugin'i Compose compiler icin kalir
- Gradle wrapper minimum `9.3.1` olmalidir
