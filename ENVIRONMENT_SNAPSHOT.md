# ENVIRONMENT_SNAPSHOT — exact environment of the published revision analyses

This file records the exact execution environment of the published revision
analyses, in particular `code/revision/wp9_cluster_inference.py`
(`results_json/revision/wp9_cluster_inference.json`). `requirements.txt`
specifies minimum versions only; results such as bootstrap confidence-interval
trailing digits can drift across library versions, so the published JSON files
are the canonical record and this snapshot documents the environment that
produced them.

## Platform

- Python: 3.13.12 (CPython)
- OS/platform: macOS 26.5.2 (arm64, Apple silicon; Darwin 25.5.0)

## Key packages

- numpy 2.3.4
- pandas 2.3.3
- scipy 1.17.0
- statsmodels 0.14.6
- scikit-learn 1.8.0

## Fixed run parameters (wp9_cluster_inference.py)

- random seed: 20260511
- paired condition-cluster bootstrap: B = 2000 (47 condition clusters)
- repeated grouped cross-validation: R = 200 fold shuffles, k = 5 grouped folds
- GEE: exchangeable working correlation attempted first; prespecified
  independence-working-correlation fallback used and recorded in the JSON
  (`working_correlation` key). In the published run the independence fallback
  converged with condition-cluster-robust standard errors.

## Full `pip freeze` of the execution environment

```
adjustText==1.3.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anthropic==0.84.0
anyio==4.12.1
appdirs==1.4.4
astroid==4.0.3
attrs==25.4.0
av==16.1.0
black==26.1.0
blinker==1.9.0
certifi==2026.1.4
cffi==2.0.0
charset-normalizer==3.4.4
click==8.4.2
contourpy==1.3.3
coverage==7.13.3
cryptography==46.0.4
ctranslate2==4.7.1
cycler==0.12.1
dash==4.1.0
dash-bootstrap-components==2.0.4
dataclasses-json==0.6.7
datasets==4.6.0
dill==0.4.0
diskcache==5.6.3
distro==1.9.0
docstring_parser==0.17.0
easyocr==1.7.2
et_xmlfile==2.0.0
faster-whisper==1.2.1
filelock==3.24.2
Flask==3.1.3
flatbuffers==25.12.19
fonttools==4.61.1
frozenlist==1.8.0
fsspec==2026.2.0
geopandas==1.1.2
greenlet==3.3.1
h11==0.16.0
hf-xet==1.5.2
httpcore==1.0.9
httpx==0.28.1
httpx-sse==0.4.3
huggingface_hub==1.24.0
hwpx==1.1.1
idna==3.11
ImageIO==2.37.2
importlib_metadata==9.0.0
iniconfig==2.3.0
instructor==1.14.5
isort==7.0.0
itsdangerous==2.2.0
Jinja2==3.1.6
jiter==0.11.1
joblib==1.5.3
jsonpatch==1.33
jsonpointer==3.0.0
kiwisolver==1.4.9
langchain==1.2.10
langchain-classic==1.0.1
langchain-community==0.4.1
langchain-core==1.2.15
langchain-ollama==1.0.1
langchain-openai==1.1.10
langchain-text-splitters==1.1.1
langgraph==1.0.9
langgraph-checkpoint==4.0.0
langgraph-prebuilt==1.0.8
langgraph-sdk==0.3.9
langsmith==0.7.6
lazy_loader==0.4
librt==0.7.8
lxml==6.0.2
markdown-it-py==4.0.0
MarkupSafe==3.0.3
marshmallow==3.26.2
matplotlib==3.10.8
mccabe==0.7.0
mdurl==0.1.2
mpmath==1.3.0
multidict==6.7.1
multiprocess==0.70.18
mypy==1.19.1
mypy_extensions==1.1.0
narwhals==2.20.0
nest-asyncio==1.6.0
networkx==3.6.1
ninja==1.13.0
numpy==2.3.4
olefile==0.47
ollama==0.6.1
onnxruntime==1.24.1
openai==2.24.0
opencv-python==4.13.0.92
opencv-python-headless==4.13.0.92
openpyxl==3.1.5
orjson==3.11.7
ormsgpack==1.12.2
packaging==26.0
pandas==2.3.3
pathspec==1.0.4
patsy==1.0.2
pdf2image==1.17.0
pdfminer.six==20251230
pdfplumber==0.11.9
pillow==12.1.0
pillow_heif==1.3.0
platformdirs==4.5.1
playwright==1.58.0
plotly==6.7.0
pluggy==1.6.0
propcache==0.4.1
protobuf==6.33.5
pyarrow==23.0.1
pyclipper==1.4.0
pycparser==3.0
pydantic==2.12.5
pydantic-settings==2.13.1
pydantic_core==2.41.5
pyee==13.0.0
Pygments==2.19.2
pylint==4.0.4
PyMuPDF==1.27.2.2
pyogrio==0.12.1
pypandoc_binary==1.17
pyparsing==3.3.2
pypdf==6.7.1
PyPDF2==3.0.1
pypdfium2==5.7.1
pyproj==3.7.2
pytest==9.0.2
pytest-base-url==2.1.0
pytest-cov==7.0.0
pytest-playwright==0.7.2
python-bidi==0.6.7
python-dateutil==2.9.0.post0
python-docx==1.2.0
python-dotenv==1.2.1
python-pptx==1.0.2
python-slugify==8.0.4
pytokens==0.4.1
pytz==2025.2
PyYAML==6.0.3
ragas==0.4.3
regex==2026.1.15
requests==2.32.5
requests-toolbelt==1.0.0
retrying==1.4.2
rich==14.3.2
safetensors==0.8.0
scikit-image==0.26.0
scikit-learn==1.8.0
scikit-network==0.33.5
scipy==1.17.0
segmentation_models_pytorch==0.5.0
setuptools==81.0.0
shapely==2.1.2
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
SQLAlchemy==2.0.47
statsmodels==0.14.6
sympy==1.14.0
tenacity==9.1.4
text-unidecode==1.3
threadpoolctl==3.6.0
tifffile==2026.1.28
tiktoken==0.12.0
timm==1.0.28
tokenizers==0.22.2
tomlkit==0.14.0
torch==2.10.0
torchvision==0.25.0
tqdm==4.67.3
transformers==5.14.0
typer==0.24.0
typer-slim==0.24.0
types-requests==2.32.4.20260107
typing-inspect==0.9.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.6.3
uuid_utils==0.14.1
Werkzeug==3.1.8
xlsxwriter==3.2.9
xxhash==3.6.0
yarl==1.22.0
yt-dlp==2026.2.4
zipp==3.23.1
zstandard==0.25.0
```
