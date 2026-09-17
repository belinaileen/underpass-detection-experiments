# underpass-detection-experiments

## Demo Areas (meeting 16 march)
Two tiles:

Weesperstraat, Amsterdam (3DBAG tile 10/434/716)
```
POLYGON ((122093.33100000000558794 485890.39699999999720603, 122593.33100000000558794 485890.39699999999720603, 122593.33100000000558794 486390.39699999999720603, 122093.33100000000558794 486390.39699999999720603, 122093.33100000000558794 485890.39699999999720603))
```

Beemsterstraat, Amsterdam (3DBAG tile 9/444/728)
```
POLYGON ((124593.33100000000558794 488890.39699999999720603, 125593.33100000000558794 488890.39699999999720603, 125593.33100000000558794 489890.39699999999720603, 124593.33100000000558794 489890.39699999999720603, 124593.33100000000558794 488890.39699999999720603))
```


## Underpass Detection in 2D

The underpass detection pipeline in 2D runs in three stages:

### 1. Underpass Detection (underpass_detection_2d)

Detects underpass geometries by comparing BAG and BGT building polygons. 

```bash
cd underpass_detection_2d
cp .env.example .env      # <----- edit with your DB credentials & the desired table names
uv pip install -e .
python scripts/detect_underpasses.py
```

Outputs: `underpasses.geometries`, `underpasses.bag_bgt_join`, `underpasses.bag_minus_bgt`, `underpasses.snapped_differences`

### 2. Edge Classification (edge-classification)

Classifies underpass polygon edges as interior, exterior, or shared.

```bash
cd edge-classification
cp .env.example .env      # edit with your DB credentials
uv pip install -e .
python scripts/classify_all_edges.py
```

Outputs: `underpasses.edges`


### 3. Edge Offset (edge-offset)

Offsets underpass polygon edges to produce extended geometries.

```bash
cd edge-offset
cp .env.example .env      # <----- edit with your DB credentials & the desired table names
uv pip install -e .
python scripts/offset_all_polygons.py
```

Outputs: `underpasses.extended_geometries`, `underpasses.skipped_underpasses`




## Rotterdam3d_underpass_extraction

This is a submodule (created by C.Moon) with the code for extracting underpasses (outer ceiling surfaces) from the 3D Rottedam and 3D Den Haag data. 

### How to install the submodule (first time clone):

If you just cloned this repository and want to initialize the submodule, run:

```
git submodule update --init --recursive
```

### How to update the submodule to the latest commit:

To update the submodule to the latest commit from its tracked branch, run:

```
git submodule update --remote --merge
```

After updating, commit the change in the main repository:

```
git add rotterdam3d_underpass_extraction
git commit -m "Update submodule to latest commit"
```


