import re

def clean_text(text: str) -> str:
    """
    Nettoyage du texte avant chunking.
    """

    if not text:
        return ""

    # URLs Reddit
    text = re.sub(r"http\S+", "", text)

    # Mentions Reddit
    text = re.sub(r"u\/\w+", "", text)

    # Espaces multiples
    text = re.sub(r"\s+", " ", text)

    # Suppression artefacts courants
    text = text.replace("nan", "")
    text = text.replace("None", "")

    return text.strip()