# Deep Learning for Super Resolution - Template 

Aques repositori és un exemple de codi per entrenar i testejar un model de deep learning per super resolució

## Configuració de l'entorn virtual

Asegura't que tens instaal·lat python3.12 y virtualenv (se pot fer amb qualsevol gestor d'entorns virtuals)

### Instal·lació de Python 3.12

1. **Instala Python 3.12**: Si no tens Python 3.12, pots instal·lar-lo seguint les instruccions del seu lloc web [Python](https://www.python.org/downloads/).
2. Assegura't d'afegir el python al teu PATH.

### Instal·lació de virtualenv
```plaintext
> pip install virtualenv
```
### Creació, activació i instal·lació dels requeriments de l'entorn virtual
```plaintext
> python3.12 -m venv venv
> source ${RUTA_DE_LENTORN_VIRTUAL}/venv/bin/activate
> pip install -r requirements.txt
```

## Dataset:

Us podeu descarregar les dades del dataset [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

En aquest exemple s'ha agafat imatges de alta resolució tallades de 512x512 i generam les imatge de baixa resolució nosaltres mateixos.
Si voleu entrenar en condicions diferents haureu d'editar l'arxiu del dataset DIV2K.py

```plaintext
dataset/
├── train/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── validation/
│   ├── img1.png
│   ├── img2.png
│   └── ...
└── test/
    ├── img1.png
    ├── img2.png
    └── ...
```
## Entrenament

Per entrenar he d'executar la següent comanda en una temrinal on tingeu l'entorn virtual activat. Suposant que esteu dintre de la carpeta TAMI-template
```plaintext
> python src/train.py --sampling 2 --dataset_path ${RUTA_DEL_DATASET}
```
Recordeu que en l'argparser hi ha molts d'arguments que podeu editar, si no teniu gpu serà necessari que li canvieu el dispositiu on s'executa tot afegint "--device cpu"

## Testeig

Per executar l'spcrip de test recordeu que és imprescindible assegurar-se que els arguments de instanciació del model siguin els mateixos que amb que heu entrenat 

L'Script de test ens demana un directori on volem guardar les imatges i la ruta als pesos.

Per defecta el logger guarda els darrers pesos i els que han assolit la millor loss ( la més baixa ) en el conjunt de validació. 

Utitzem els pesos del best per obtenir les mètriques que millor funcionen en validació.
Els pesos es guardaran a la següent ruta:

```plaintext
CKPT_PATH="./logs/DIV2K/SRNet/YYYY-MM-DD/${NICKNAME}/checkpoints/best.ckpt"
```
o bé,
```plaintext
CKPT_PATH="./logs/DIV2K/SRNet/YYYY-MM-DD/checkpoints/best.ckpt",
```
si no heu introduit cap nickname durant l'entrenament.

Example d'execució del testing
```plaintext
> python src/test.py --ckpt_path ${CKPT_PATH} --output_path ${IMAGE_DIR} --sampling 2 --dataset_path ${RUTA_DEL_DATASET}
```


