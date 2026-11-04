info = {
        "download_reference": "peluz/lener_br",
        "splits": {"train": "./src/data/lener/train.conll", 
                   "val": "./src/data/lener/dev.conll", 
                   "test": "./src/data/lener/test.conll"},
        "n_labels": 13,
        "HOMEPAGE": "https://cic.unb.br/~teodecampos/LeNER-Br/",
        "URL": "https://github.com/peluz/lener-br/raw/master/leNER-Br/",
        "TRAINING_FILE": "train.conll",
        "DEV_FILE": "dev.conll",
        "TEST_FILE": "test.conll",
        "labels": ["O", "B-ORGANIZACAO", "I-ORGANIZACAO", "B-PESSOA","I-PESSOA", "B-TEMPO", "I-TEMPO",
                    "B-LOCAL", "I-LOCAL", "B-LEGISLACAO", "I-LEGISLACAO", "B-JURISPRUDENCIA", "I-JURISPRUDENCIA"],
        "label2id": {"O":0, "B-ORGANIZACAO":1, "I-ORGANIZACAO":2, "B-PESSOA":3, "I-PESSOA":4, "B-TEMPO":5, 
                     "I-TEMPO":6, "B-LOCAL":7, "I-LOCAL":8, "B-LEGISLACAO":9, "I-LEGISLACAO":10, 
                     "B-JURISPRUDENCIA":11, "I-JURISPRUDENCIA":12},
        "id2label": {"0":"O", "1":"B-ORGANIZACAO", "2":"I-ORGANIZACAO", "3":"B-PESSOA", "4":"I-PESSOA", "5":"B-TEMPO",
                     "6":"I-TEMPO", "7":"B-LOCAL", "8":"I-LOCAL", "9":"B-LEGISLACAO", "10":"I-LEGISLACAO",
                     "11":"B-JURISPRUDENCIA", "12":"I-JURISPRUDENCIA"}
    }

from pathlib import Path

# Used by scripts/eval_lener_br.py and scripts/train_lener_br.py.
DATA_DIR = Path(__file__).resolve().parent

SPLIT_FILES = {"train": "train.conll", "dev": "dev.conll", "test": "test.conll"}

# LeNER-Br entity types (CoNLL tag without the B-/I- prefix) with short
# Portuguese descriptions, per the tutorial's guidance that descriptions
# improve extraction accuracy.
ENTITY_DESCRIPTIONS = {
    "ORGANIZACAO": "Nomes de organizações, instituições, empresas ou órgãos públicos",
    "PESSOA": "Nomes de pessoas",
    "TEMPO": "Expressões temporais como datas, horas, prazos ou períodos",
    "LOCAL": "Nomes de lugares, cidades, estados ou localizações geográficas",
    "LEGISLACAO": "Referências a leis, decretos, normas ou outros dispositivos legais",
    "JURISPRUDENCIA": "Referências a decisões judiciais, acórdãos ou jurisprudência",
}

# Entity type name the GLiNER2 model is queried/trained with for each CoNLL tag.
# Must equal the accent-free, lowercase form of the labels in conll_to_gliner2.py.
ENTITY_NAMES = {tag: tag.lower() for tag in ENTITY_DESCRIPTIONS}

from datasets import load_dataset
def get_raw_split(split):
    raw_ds = load_dataset(info['download_reference'], trust_remote_code=True)
     # lener train_dataset is a map-style dataset
    return raw_ds#['validation']