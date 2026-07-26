# Fonts

Place a bold, high-legibility `.ttf` font here for captions — the classic
"Hormozi style" look uses a heavy sans-serif such as:

- Montserrat ExtraBold / Black (free on Google Fonts)
- Komika Axis
- Obelix Pro
- Archivo Black

Download example (Montserrat, free/open license) and place it at:

```
assets/fonts/Montserrat-ExtraBold.ttf
```

Then make sure `CAPTION_FONT_PATH` in your `.env` points to the same path.
If no font is found at that path, `modules/editor.py` will fall back to a
plain default font and log a warning — captions will still render, just less
stylized.
