# Image Captioning Gallery

A web-based gallery that can generates image captioning for uploaded images.

The image caption is computed at the time of the upload, and saved to .json file
for display later. 

## How to install?

### Docker

```bash
docker compose build
```

### Manually

```bash
pip install -r requirements.txt
```

## How to run?

### Docker

```bash
docker compose up
```

### Manually

```bash
python app.py
```