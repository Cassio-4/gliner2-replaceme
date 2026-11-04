info = {
        "download_reference": "https://github.com/ulysses-camara/ulysses-ner-br/tree/main/PL-corpus_v2/ulysses_categories/holdout",
        "splits": {"train": "./src/data/ulyssesbr/pl_corpus_v2_categories/train.txt",
                   "val": "./src/data/ulyssesbr/pl_corpus_v2_categories/dev.txt", 
                   "test": "./src/data/ulyssesbr/pl_corpus_v2_categories/test.txt"},
        "n_labels": 15,
        "HOMEPAGE": "https://cic.unb.br/~teodecampos/LeNER-Br/",
        "URL": "https://github.com/ulysses-camara/ulysses-ner-br/tree/main",
        "labels": ['B-DATA', 'B-EVENTO', 'B-FUNDAMENTO', 'B-LOCAL', 'B-ORGANIZACAO', 'B-PESSOA', 'B-PRODUTODELEI', 
                   'I-DATA', 'I-EVENTO', 'I-FUNDAMENTO', 'I-LOCAL', 'I-ORGANIZACAO', 'I-PESSOA', 'I-PRODUTODELEI', 'O'],
        "label2id": {'B-DATA':0, 'B-EVENTO':1, 'B-FUNDAMENTO':2, 'B-LOCAL':3, 'B-ORGANIZACAO':4, 'B-PESSOA':5, 
                     'B-PRODUTODELEI':6, 'I-DATA':7, 'I-EVENTO':8, 'I-FUNDAMENTO':9, 'I-LOCAL':10, 'I-ORGANIZACAO':11, 
                     'I-PESSOA':12, 'I-PRODUTODELEI':13, 'O':14},
        "id2label": {'0':'B-DATA', '1':'B-EVENTO', '2':'B-FUNDAMENTO', '3':'B-LOCAL', '4':'B-ORGANIZACAO', '5':'B-PESSOA', 
                     '6':'B-PRODUTODELEI', '7':'I-DATA', '8':'I-EVENTO', '9':'I-FUNDAMENTO', '10':'I-LOCAL', '11':'I-ORGANIZACAO', 
                     '12':'I-PESSOA', '13':'I-PRODUTODELEI', '14':'O'}
    }

from pathlib import Path

# Used by scripts/eval_lener_br.py and scripts/train_lener_br.py.
DATA_DIR = Path(__file__).resolve().parent

SPLIT_FILES = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}

# UlyssesNER-Br (PL-corpus v2, categories) entity types (CoNLL tag without the
# B-/I- prefix) with short Portuguese descriptions.
ENTITY_DESCRIPTIONS = {
    "DATA": "Datas ou referências temporais, como dias, meses e anos",
    "EVENTO": "Nomes de eventos, como eleições, congressos, competições esportivas ou crises",
    "FUNDAMENTO": "Referências a fundamentos legais, como leis, decretos, códigos, estatutos ou artigos de lei",
    "LOCAL": "Nomes de lugares, cidades, estados, países ou localizações geográficas",
    "ORGANIZACAO": "Nomes de organizações, instituições, empresas ou órgãos públicos",
    "PESSOA": "Nomes de pessoas",
    "PRODUTODELEI": "Produtos de lei, como programas, fundos, sistemas, regimes ou políticas públicas criados por lei",
}

# Entity type name the GLiNER2 model is queried/trained with for each CoNLL tag.
# Must equal the accent-free, lowercase form of the labels in conll_to_gliner2.py.
ENTITY_NAMES = {
    "DATA": "data",
    "EVENTO": "evento",
    "FUNDAMENTO": "fundamento legal",
    "LOCAL": "local",
    "ORGANIZACAO": "organizacao",
    "PESSOA": "pessoa",
    "PRODUTODELEI": "produto de lei",
}