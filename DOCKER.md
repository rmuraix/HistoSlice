# Docker Examples for HistoSlice

## Running the CLI

### Basic usage with mounted volumes
```bash
docker run --rm \
  -v $(pwd)/slides:/data/input \
  -v $(pwd)/output:/data/output \
  ghcr.io/rmuraix/histoslice:latest \
  --input '/data/input/*.tiff' \
  --output /data/output \
  --width 512 \
  --overlap 0.5 \
  --max-background 0.5 \
  --thumbnails \
  --metrics
```

### Using specific version tag
```bash
docker run --rm \
  -v $(pwd)/slides:/data/input \
  -v $(pwd)/output:/data/output \
  ghcr.io/rmuraix/histoslice:0.5.0 \
  --input '/data/input/*.tiff' \
  --output /data/output \
  --width 512
```

### Get help
```bash
docker run --rm ghcr.io/rmuraix/histoslice:latest --help
```

## Using the Python API

### Interactive Python shell
```bash
docker run --rm -it \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/rmuraix/histoslice:latest \
  python
```

Then in the Python shell:
```python
from histoslice import SlideReader
reader = SlideReader("./slides/slide.jpeg")
threshold, tissue_mask = reader.get_tissue_mask(level=-1)
```

### Running a Python script
Create a script `process_slides.py`:
```python
from histoslice import SlideReader

reader = SlideReader("/workspace/slides/slide.jpeg")
threshold, tissue_mask = reader.get_tissue_mask(level=-1)
tile_coordinates = reader.get_tile_coordinates(
    tissue_mask, width=512, overlap=0.5, max_background=0.5
)
tile_metadata, failures = reader.save_regions(
    "/workspace/output/",
    tile_coordinates,
    threshold=threshold,
    save_metrics=True,
    save_thumbnail=True
)
print(f"Saved {len(tile_metadata)} tiles")
if failures:
    print(f"Failed tiles: {len(failures)}")
```

Run it with:
```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/rmuraix/histoslice:latest \
  python process_slides.py
```

## Using Docker Compose

Create a `docker-compose.yml`:
```yaml
services:
  histoslice:
    image: ghcr.io/rmuraix/histoslice:latest
    volumes:
      - ./slides:/data/input
      - ./output:/data/output
    command: --input '/data/input/*.tiff' --output /data/output --width 512 --overlap 0.5 --thumbnails --metrics
```

Then run:
```bash
docker-compose up
```

## Building the Image Locally

```bash
# Build
docker build -t histoslice:local .

# Run
docker run --rm \
  -v $(pwd)/slides:/data/input \
  -v $(pwd)/output:/data/output \
  histoslice:local \
  --input '/data/input/*.tiff' \
  --output /data/output \
  --width 512
```

## Notes

- The container uses Python 3.12 with UV package manager
- OpenCV is installed in headless mode (no GUI dependencies)
- All dependencies are pre-installed for immediate use
- The `/data/input` and `/data/output` directories are created by default
- Mount your local directories to these paths for easy file access
