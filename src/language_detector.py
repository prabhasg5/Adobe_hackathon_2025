from lingua import Language, LanguageDetectorBuilder

def detect_languages(lines):
    detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
    lang_map = {}
    for i, line in enumerate(lines):
        text = line["text"]
        if text.strip():
            lang = detector.detect_language_of(text)
            lang_map[i] = lang.iso_code_639_1.name if lang else "UNKNOWN"
        else:
            lang_map[i] = "UNKNOWN"
    return lang_map



