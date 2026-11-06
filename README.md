# Multi MA/AP Detector MT5 Backtest

MA ve AP (average price) temelli sinyalleri geçmiş piyasa verisi üzerinde inceleyen bir Python backtest aracıdır. MetaTrader 5 terminalinden kapanmış mum verilerini alabilir veya CSV dosyalarıyla çalışabilir. **Gerçek emir göndermez; canlı işlem botu değildir.**

## Gereksinimler

- Python 3.10 veya üzeri
- `numpy` ve `pandas`
- MT5'ten veri çekmek için Windows'ta kurulu/açık MetaTrader 5 terminali ve `MetaTrader5` Python paketi

Kurulum:

```bash
python -m pip install -r requirements.txt
```

MT5 Python paketi kullanacağınız Python/MetaTrader ortamında ayrıca kurulu olmalıdır:

```bash
python -m pip install MetaTrader5
```

## Kullanım

MT5'ten varsayılan üç zaman dilimi motoruyla backtest:

```bash
python Multi_MA_Detector_MT5_Backtest.py --portfolio
```

CSV ile tek motor backtest:

```bash
python Multi_MA_Detector_MT5_Backtest.py --single --csv bars.csv
```

CSV dosyasında `time,open,high,low,close,volume` sütunları bulunmalıdır. Zaman damgaları UTC olmalı; fiyatlar BID OHLC olmalıdır. MTF CSV çalışması için `--mtf --higher-csv higher_bars.csv` seçeneklerini kullanın. Yapılandırma, tüm seçenekler ve portföy CSV klasörü (`M5.csv`, `M15.csv`, `H1.csv`, `H4.csv`, `D1.csv`) hakkında yardım:

```bash
python Multi_MA_Detector_MT5_Backtest.py --help
```

Raporlar varsayılan olarak `backtest_results/` klasörüne yazılır. Bu klasör ve ham piyasa verileri Git'e eklenmez.

## Backtest varsayımları

- Sinyaller kapanmış mumları kullanır; giriş bir sonraki mum açılışında varsayılır.
- Belirsiz mum içi fiyat sırası için iki OHLC yolu da simüle edilir ve daha düşük kapanış sermayesi sonucu tutulur. Bu, tick bazlı simülasyon değildir.
- Spread, kayma, komisyon, taşıma maliyeti ve PnL dönüşümü varsayımları sonucu etkiler. Geçmiş kur dönüşümü ve teminat/stop-out modellenmez.
- Backtest sonucu gelecekteki performansın göstergesi değildir; parametre araması veya kârlılık garantisi sunmaz.

## Lisans

Henüz lisans seçilmedi. Yeniden kullanım koşullarını belirlemek için GitHub'a yüklemeden önce uygun bir lisans ekleyin (ör. MIT).
