import re
import pandas as pd

INPUT_FILE  = 'data/processed/splits/test.csv'
OUTPUT_FILE = 'data/processed/splits/test_obfuscated_homograph.csv'
TEXT_COL    = 'text'
SUBJECT_COL = 'subject'    # set to None to skip subject obfuscation

PHISHING_KEYWORDS = [
    'verify', 'verified', 'verification',
    'account', 'accounts',
    'password', 'passwords',
    'click',
    'suspend', 'suspended', 'suspension',
    'update', 'updated', 'updates',
    'confirm', 'confirmed', 'confirmation',
    'login',
    'secure', 'security',
    'urgent', 'urgently',
    'bank', 'banking',
    'credentials', 'credential',
    'access',
    'validate', 'validation',
    'immediately', 'immediate',
]

# ASCII → visually identical Unicode lookalike
HOMOGRAPH_MAP = {
    'a': 'а',  # Cyrillic а (U+0430)
    'e': 'е',  # Cyrillic е (U+0435)
    'o': 'о',  # Cyrillic о (U+043E)
    'p': 'р',  # Cyrillic р (U+0440)
    'c': 'с',  # Cyrillic с (U+0441)
    'x': 'х',  # Cyrillic х (U+0445)
    'i': 'і',  # Cyrillic і (U+0456)
}

def apply_homograph(word):
    for i, char in enumerate(word.lower()):
        if char in HOMOGRAPH_MAP:
            return word[:i] + HOMOGRAPH_MAP[char] + word[i+1:]
    return word

def homograph_attack(text):
    for keyword in PHISHING_KEYWORDS:
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        text = pattern.sub(lambda m: apply_homograph(m.group(0)), text)
    return text

if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE)
    print(f'Loaded {len(df)} emails  (legitimate={( df["label"]==0).sum()}  phishing={(df["label"]==1).sum()})')

    df_out = df.copy()
    phishing_mask = df_out['label'] == 1

    df_out.loc[phishing_mask, TEXT_COL] = (df_out.loc[phishing_mask, TEXT_COL].fillna('').apply(homograph_attack))

    if SUBJECT_COL and SUBJECT_COL in df_out.columns:
        df_out.loc[phishing_mask, SUBJECT_COL] = (df_out.loc[phishing_mask, SUBJECT_COL].fillna('').apply(homograph_attack))

    # How many phishing emails were actually modified
    modified = (df_out.loc[phishing_mask, TEXT_COL] != df.loc[phishing_mask, TEXT_COL]).sum()
    print(f'Phishing emails modified: {modified}/{phishing_mask.sum()}')

    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f'Saved → {OUTPUT_FILE}')